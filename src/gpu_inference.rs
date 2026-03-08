//! GPU inference for QORA-Vision.
//!
//! ViT forward passes using Cortex GPU tensors.
//! Key differences from QOR3B:
//!   - LayerNorm (not RMSNorm) with bias
//!   - Bidirectional attention (no causal mask)
//!   - No KV cache (full sequence processed at once)
//!   - GELU-tanh activation (not SiLU)

use cortex::prelude::*;
use cortex::tensor::activation;
use std::time::Instant;

use crate::gpu_loader::*;
use crate::siglip::{SigLIPVisionWeights, SigLIPTextWeights};
use crate::vivit::ViViTWeights;

// ============================================================
// Helper ops
// ============================================================

/// LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
fn layer_norm<B: Backend>(x: Tensor<B, 2>, gamma: &Tensor<B, 1>, beta: &Tensor<B, 1>) -> Tensor<B, 2> {
    let eps = 1e-6;
    let mean = x.clone().mean_dim(1); // [seq, 1]
    let centered = x - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(1); // [seq, 1]
    let normed = centered / (var + eps).sqrt();
    normed * gamma.clone().unsqueeze::<2>() + beta.clone().unsqueeze::<2>()
}

/// Bidirectional multi-head attention (no causal mask).
/// q, k, v: [seq_len, num_heads * head_dim]
fn mha_bidirectional<B: Backend>(
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    num_heads: usize,
    head_dim: usize,
) -> Tensor<B, 2> {
    let seq_len = q.dims()[0];
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Reshape: [seq, num_heads, head_dim] -> [num_heads, seq, head_dim]
    let q = q.reshape([seq_len, num_heads, head_dim]).swap_dims(0, 1);
    let k = k.reshape([seq_len, num_heads, head_dim]).swap_dims(0, 1);
    let v = v.reshape([seq_len, num_heads, head_dim]).swap_dims(0, 1);

    // Scores: [num_heads, seq, seq]
    let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale);

    // Softmax (no causal mask — bidirectional)
    let attn_weights = activation::softmax(scores, 2);

    // Weighted sum: [num_heads, seq, head_dim]
    let out = attn_weights.matmul(v);

    // [num_heads, seq, head_dim] -> [seq, num_heads * head_dim]
    out.swap_dims(0, 1).reshape([seq_len, num_heads * head_dim])
}

/// Cross-attention for MAP pooling (1 query, seq_len keys).
/// q: [1, num_heads * head_dim]
/// k, v: [seq_len, num_heads * head_dim]
fn cross_attention_single_query<B: Backend>(
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    num_heads: usize,
    head_dim: usize,
) -> Tensor<B, 2> {
    let seq_len = k.dims()[0];
    let scale = 1.0 / (head_dim as f32).sqrt();

    // q: [1, heads, dim] -> [heads, 1, dim]
    let q = q.reshape([1, num_heads, head_dim]).swap_dims(0, 1);
    // k,v: [seq, heads, dim] -> [heads, seq, dim]
    let k = k.reshape([seq_len, num_heads, head_dim]).swap_dims(0, 1);
    let v = v.reshape([seq_len, num_heads, head_dim]).swap_dims(0, 1);

    // Scores: [heads, 1, seq]
    let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale);
    let attn_weights = activation::softmax(scores, 2);

    // Out: [heads, 1, dim]
    let out = attn_weights.matmul(v);

    // [heads, 1, dim] -> [1, heads * dim]
    out.swap_dims(0, 1).reshape([1, num_heads * head_dim])
}

// ============================================================
// ViT layer forward (shared)
// ============================================================

fn vit_layer_forward_gpu<B: Backend>(
    x: Tensor<B, 2>,
    layer: &GpuViTLayer<B>,
    num_heads: usize,
    head_dim: usize,
) -> Tensor<B, 2> {
    // Pre-norm + Self-Attention
    let x_norm = layer_norm(x.clone(), &layer.ln1_gamma, &layer.ln1_beta);

    // QKV projections: [seq, hidden] @ [hidden, hidden] + bias
    let q = x_norm.clone().matmul(layer.q_proj.clone()) + layer.q_bias.clone().unsqueeze::<2>();
    let k = x_norm.clone().matmul(layer.k_proj.clone()) + layer.k_bias.clone().unsqueeze::<2>();
    let v = x_norm.matmul(layer.v_proj.clone()) + layer.v_bias.clone().unsqueeze::<2>();

    // Multi-head attention (bidirectional, no mask)
    let attn_out = mha_bidirectional(q, k, v, num_heads, head_dim);

    // Output projection + residual
    let attn_out = attn_out.matmul(layer.o_proj.clone()) + layer.o_bias.clone().unsqueeze::<2>();
    let h = x + attn_out;

    // Pre-norm + MLP
    let h_norm = layer_norm(h.clone(), &layer.ln2_gamma, &layer.ln2_beta);

    // fc1 + gelu_tanh + fc2
    let fc1_out = h_norm.matmul(layer.mlp_fc1.clone()) + layer.mlp_fc1_bias.clone().unsqueeze::<2>();
    let fc1_act = activation::gelu(fc1_out);
    let fc2_out = fc1_act.matmul(layer.mlp_fc2.clone()) + layer.mlp_fc2_bias.clone().unsqueeze::<2>();

    // Residual
    h + fc2_out
}

// ============================================================
// SigLIP Vision GPU forward
// ============================================================

pub fn siglip_vision_forward_gpu<B: Backend>(
    image: &[f32],
    cpu_weights: &SigLIPVisionWeights,
    gpu_weights: &GpuSigLIPVision<B>,
) -> Vec<f32> {
    let device = gpu_weights.patch_embed_weight.device();
    let num_patches = gpu_weights.num_patches;
    let num_heads = gpu_weights.num_heads;
    let head_dim = gpu_weights.head_dim;

    // 1. Patch extraction on CPU (same as siglip.rs patch_embed)
    eprintln!("  Patch embedding (GPU)...");
    let ps = cpu_weights.patch_size;
    let nc = cpu_weights.num_channels;
    let img_size = 224;
    let grid = img_size / ps;
    let patch_dim = nc * ps * ps;

    let mut patches = vec![0.0f32; num_patches * patch_dim];
    for py in 0..grid {
        for px in 0..grid {
            let patch_idx = py * grid + px;
            for c in 0..nc {
                for dy in 0..ps {
                    for dx in 0..ps {
                        let img_y = py * ps + dy;
                        let img_x = px * ps + dx;
                        let src = c * img_size * img_size + img_y * img_size + img_x;
                        let dst = c * ps * ps + dy * ps + dx;
                        patches[patch_idx * patch_dim + dst] = image[src];
                    }
                }
            }
        }
    }

    // Upload patches to GPU and project
    let patches_gpu: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(patches, [num_patches, patch_dim]),
        &device,
    );

    // [num_patches, patch_dim] @ [patch_dim, hidden] + bias
    // weight is [hidden, patch_dim], so transpose
    let embeddings = patches_gpu.matmul(gpu_weights.patch_embed_weight.clone().transpose())
        + gpu_weights.patch_embed_bias.clone().unsqueeze::<2>();

    // 2. Add position embeddings
    let mut x = embeddings + gpu_weights.position_embedding.clone();

    // 3. ViT backbone
    eprintln!("  ViT backbone ({} layers, GPU)...", gpu_weights.layers.len());
    for (i, layer) in gpu_weights.layers.iter().enumerate() {
        if i % 4 == 0 {
            eprintln!("    Layer {i}/{}...", gpu_weights.layers.len());
        }
        x = vit_layer_forward_gpu(x, layer, num_heads, head_dim);
    }

    // 4. Post-LayerNorm
    let x = layer_norm(x, &gpu_weights.post_ln_gamma, &gpu_weights.post_ln_beta);

    // 5. MAP pooling
    eprintln!("  MAP pooling (GPU)...");

    // Query from probe: [1, hidden]
    let probe: Tensor<B, 2> = gpu_weights.map_probe.clone().unsqueeze::<2>(); // [1, hidden]
    let q = probe.matmul(gpu_weights.map_q_weight.clone()) + gpu_weights.map_q_bias.clone().unsqueeze::<2>();

    // K, V from encoder output
    let k = x.clone().matmul(gpu_weights.map_k_weight.clone()) + gpu_weights.map_k_bias.clone().unsqueeze::<2>();
    let v = x.matmul(gpu_weights.map_v_weight.clone()) + gpu_weights.map_v_bias.clone().unsqueeze::<2>();

    // Cross-attention: 1 query, num_patches keys
    let attn_out = cross_attention_single_query(q, k, v, num_heads, head_dim);

    // Output projection
    let projected = attn_out.matmul(gpu_weights.map_out_weight.clone()) + gpu_weights.map_out_bias.clone().unsqueeze::<2>();

    // Residual + LayerNorm + MLP
    let residual = projected;
    let normed = layer_norm(residual.clone(), &gpu_weights.map_ln_gamma, &gpu_weights.map_ln_beta);
    let fc1 = normed.matmul(gpu_weights.map_mlp_fc1.clone()) + gpu_weights.map_mlp_fc1_bias.clone().unsqueeze::<2>();
    let fc1_act = activation::gelu(fc1);
    let fc2 = fc1_act.matmul(gpu_weights.map_mlp_fc2.clone()) + gpu_weights.map_mlp_fc2_bias.clone().unsqueeze::<2>();
    let output = residual + fc2;

    // Return as CPU vec
    output.to_data().to_vec::<f32>().unwrap()
}

// ============================================================
// SigLIP Text GPU forward
// ============================================================

pub fn siglip_text_forward_gpu<B: Backend>(
    token_ids: &[u32],
    cpu_weights: &SigLIPTextWeights,
    gpu_weights: &GpuSigLIPText<B>,
) -> Vec<f32> {
    let device = gpu_weights.position_embedding.device();
    let num_heads = gpu_weights.num_heads;
    let head_dim = gpu_weights.head_dim;
    let hidden = gpu_weights.hidden_size;
    let seq_len = token_ids.len();

    // 1. Token embedding: lookup on CPU (Q4 select not supported on GPU), upload to GPU
    let mut embed_data = vec![0.0f32; seq_len * hidden];
    for (t, &tid) in token_ids.iter().enumerate() {
        let row = crate::gemv::embed_lookup(&cpu_weights.token_embedding, tid as usize, hidden);
        embed_data[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }
    let token_embeds: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(embed_data, [seq_len, hidden]),
        &device,
    );

    // Position embedding (slice to seq_len)
    let pos_embeds = gpu_weights.position_embedding.clone().slice([0..seq_len, 0..hidden]);

    let mut x = token_embeds + pos_embeds;

    // 2. ViT backbone
    for layer in &gpu_weights.layers {
        x = vit_layer_forward_gpu(x, layer, num_heads, head_dim);
    }

    // 3. Last token -> LayerNorm -> head
    let last_token = x.slice([seq_len - 1..seq_len, 0..hidden]);
    let normed = layer_norm(last_token, &gpu_weights.final_ln_gamma, &gpu_weights.final_ln_beta);
    let output = normed.matmul(gpu_weights.head_weight.clone()) + gpu_weights.head_bias.clone().unsqueeze::<2>();

    output.to_data().to_vec::<f32>().unwrap()
}

// ============================================================
// ViViT GPU forward
// ============================================================

pub fn vivit_forward_gpu<B: Backend>(
    video: &[f32],
    cpu_weights: &ViViTWeights,
    gpu_weights: &GpuViViT<B>,
) -> (Vec<f32>, Vec<f32>) {
    let device = gpu_weights.tubelet_weight.device();
    let hidden = gpu_weights.hidden_size;
    let num_heads = gpu_weights.num_heads;
    let head_dim = gpu_weights.head_dim;
    let num_patches = gpu_weights.num_patches;
    let seq_len = 1 + num_patches;

    // 1. Tubelet extraction on CPU (same as vivit.rs)
    eprintln!("  Tubelet embedding ({num_patches} patches, GPU)...");
    let [tt, th, tw] = cpu_weights.tubelet_size;
    let nc = 3usize;
    let nf = cpu_weights.num_frames;
    let img_size = 224usize;
    let grid_t = nf / tt;
    let grid_h = img_size / th;
    let grid_w = img_size / tw;
    let tubelet_dim = nc * tt * th * tw;

    let mut tubelets = vec![0.0f32; num_patches * tubelet_dim];
    for pt in 0..grid_t {
        for py in 0..grid_h {
            for px in 0..grid_w {
                let patch_idx = pt * grid_h * grid_w + py * grid_w + px;
                for c in 0..nc {
                    for dt in 0..tt {
                        for dy in 0..th {
                            for dx in 0..tw {
                                let frame = pt * tt + dt;
                                let img_y = py * th + dy;
                                let img_x = px * tw + dx;
                                let src = c * nf * img_size * img_size +
                                          frame * img_size * img_size +
                                          img_y * img_size + img_x;
                                let dst = c * tt * th * tw + dt * th * tw + dy * tw + dx;
                                tubelets[patch_idx * tubelet_dim + dst] = video[src];
                            }
                        }
                    }
                }
            }
        }
    }

    // Upload and project tubelets
    let tubelets_gpu: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(tubelets, [num_patches, tubelet_dim]),
        &device,
    );

    // [num_patches, tubelet_dim] @ [tubelet_dim, hidden] + bias
    let patch_embeds = tubelets_gpu.matmul(gpu_weights.tubelet_weight.clone().transpose())
        + gpu_weights.tubelet_bias.clone().unsqueeze::<2>();

    // 2. Prepend CLS token + position embeddings
    // CLS: [1, hidden]
    let cls: Tensor<B, 2> = gpu_weights.cls_token.clone().unsqueeze::<2>();

    // Concatenate: [1, hidden] ++ [num_patches, hidden] = [seq_len, hidden]
    let embeddings = Tensor::cat(vec![cls, patch_embeds], 0);

    // Add position embeddings
    let mut x = embeddings + gpu_weights.position_embedding.clone();

    // 3. ViT backbone
    eprintln!("  ViT backbone ({} layers, seq_len={}, GPU)...", gpu_weights.layers.len(), seq_len);
    for (i, layer) in gpu_weights.layers.iter().enumerate() {
        if i % 4 == 0 {
            eprintln!("    Layer {i}/{}...", gpu_weights.layers.len());
        }
        x = vit_layer_forward_gpu(x, layer, num_heads, head_dim);
    }

    // 4. CLS token -> LayerNorm -> classifier
    let cls_hidden = x.slice([0..1, 0..hidden]);
    let cls_normed = layer_norm(cls_hidden, &gpu_weights.final_ln_gamma, &gpu_weights.final_ln_beta);

    let logits = cls_normed.clone().matmul(gpu_weights.classifier_weight.clone())
        + gpu_weights.classifier_bias.clone().unsqueeze::<2>();

    let embedding = cls_normed.to_data().to_vec::<f32>().unwrap();
    let logits = logits.to_data().to_vec::<f32>().unwrap();

    (embedding, logits)
}

// ============================================================
// Top-level GPU entry points (with 128MB stack + catch_unwind)
// ============================================================

fn panic_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        format!("GPU panic: {s}")
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        format!("GPU panic: {s}")
    } else {
        "GPU panic: unknown error (likely out of VRAM)".to_string()
    }
}

/// Run SigLIP on GPU. Returns (image_embed, scores) or error.
pub fn run_siglip_gpu(
    image: &[f32],
    vision_weights: &SigLIPVisionWeights,
    text_weights: Option<&SigLIPTextWeights>,
    labels: &[&str],
    tokenizer: Option<&crate::tokenizer::TextTokenizer>,
    logit_scale: f32,
    logit_bias: f32,
) -> Result<(Vec<f32>, Vec<(String, f32)>), String> {
    let labels: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
    let result = std::cell::RefCell::new(None);

    std::thread::scope(|s| {
        let builder = std::thread::Builder::new()
            .name("gpu-siglip".into())
            .stack_size(128 * 1024 * 1024);
        let handle = builder.spawn_scoped(s, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_siglip_gpu_inner(image, vision_weights, text_weights, &labels,
                    tokenizer, logit_scale, logit_bias)
            }))
        });
        match handle {
            Ok(h) => {
                *result.borrow_mut() = Some(match h.join() {
                    Ok(Ok(inner)) => inner,
                    Ok(Err(panic)) => Err(panic_to_string(panic)),
                    Err(_) => Err("GPU thread panicked unexpectedly".to_string()),
                });
            }
            Err(e) => *result.borrow_mut() = Some(Err(format!("Failed to spawn GPU thread: {e}"))),
        }
    });

    result.into_inner().unwrap_or(Err("No result from GPU thread".into()))
}

fn run_siglip_gpu_inner(
    image: &[f32],
    vision_weights: &SigLIPVisionWeights,
    text_weights: Option<&SigLIPTextWeights>,
    labels: &[String],
    tokenizer: Option<&crate::tokenizer::TextTokenizer>,
    logit_scale: f32,
    logit_bias: f32,
) -> Result<(Vec<f32>, Vec<(String, f32)>), String> {
    type B = cortex::backend::Wgpu;
    let device: <B as Backend>::Device = Default::default();

    let _test: Tensor<B, 1> = Tensor::zeros([1], &device);
    eprintln!("GPU initialized");

    let t0 = Instant::now();
    let gpu_vision = load_siglip_vision_gpu::<B>(vision_weights, &device);
    eprintln!("Vision loaded to GPU in {:.1?}", t0.elapsed());

    let t_fwd = Instant::now();
    let mut image_embed = siglip_vision_forward_gpu::<B>(image, vision_weights, &gpu_vision);
    eprintln!("Vision forward (GPU) in {:.1?}", t_fwd.elapsed());

    crate::siglip::l2_normalize(&mut image_embed);

    let mut scores = Vec::new();
    if let (Some(tw), Some(tok)) = (text_weights, tokenizer) {
        let t0 = Instant::now();
        let gpu_text = load_siglip_text_gpu::<B>(tw, &device);
        eprintln!("Text loaded to GPU in {:.1?}", t0.elapsed());

        for label in labels {
            let prompt = format!("This is a photo of {label}.");
            let token_ids = tok.encode(&prompt);
            let t_fwd = Instant::now();
            let mut text_embed = siglip_text_forward_gpu::<B>(&token_ids, tw, &gpu_text);
            crate::siglip::l2_normalize(&mut text_embed);
            let score = crate::siglip::siglip_score(&image_embed, &text_embed, logit_scale, logit_bias);
            eprintln!("  {label}: {score:.4} ({:.1?})", t_fwd.elapsed());
            scores.push((label.clone(), score));
        }
    }

    Ok((image_embed, scores))
}

/// Run ViViT on GPU. Returns (embedding, logits) or error.
pub fn run_vivit_gpu(
    video: &[f32],
    weights: &ViViTWeights,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let result = std::cell::RefCell::new(None);

    std::thread::scope(|s| {
        let builder = std::thread::Builder::new()
            .name("gpu-vivit".into())
            .stack_size(128 * 1024 * 1024);
        let handle = builder.spawn_scoped(s, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_vivit_gpu_inner(video, weights)
            }))
        });
        match handle {
            Ok(h) => {
                *result.borrow_mut() = Some(match h.join() {
                    Ok(Ok(inner)) => inner,
                    Ok(Err(panic)) => Err(panic_to_string(panic)),
                    Err(_) => Err("GPU thread panicked unexpectedly".to_string()),
                });
            }
            Err(e) => *result.borrow_mut() = Some(Err(format!("Failed to spawn GPU thread: {e}"))),
        }
    });

    result.into_inner().unwrap_or(Err("No result from GPU thread".into()))
}

fn run_vivit_gpu_inner(
    video: &[f32],
    weights: &ViViTWeights,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    type B = cortex::backend::Wgpu;
    let device: <B as Backend>::Device = Default::default();

    let _test: Tensor<B, 1> = Tensor::zeros([1], &device);
    eprintln!("GPU initialized");

    let t0 = Instant::now();
    let gpu_weights = load_vivit_gpu::<B>(weights, &device);
    eprintln!("ViViT loaded to GPU in {:.1?}", t0.elapsed());

    let t_fwd = Instant::now();
    let (embedding, logits) = vivit_forward_gpu::<B>(video, weights, &gpu_weights);
    eprintln!("ViViT forward (GPU) in {:.1?}", t_fwd.elapsed());

    Ok((embedding, logits))
}

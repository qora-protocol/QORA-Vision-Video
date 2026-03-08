//! SigLIP 2 vision + text encoder.
//!
//! Vision: Conv2d patch embed → ViT → post-LN → MAP pooling → 768-dim
//! Text:   Token embed → ViT → final-LN → last token → head → 768-dim
//! Score:  sigmoid(cosine_sim * exp(logit_scale) + logit_bias)

use half::f16;
use crate::gemv::{self, Weight};
use crate::vit::ViTBackbone;

// ============================================================
// SigLIP Vision Weights
// ============================================================

pub struct SigLIPVisionWeights {
    // Patch embedding: Conv2d(3, 768, k=16, s=16) — stored as [768, 3*16*16]
    pub patch_embed_weight: Vec<f32>,  // [768, 768] (768 = 3*16*16)
    pub patch_embed_bias: Vec<f32>,    // [768]

    // Position embeddings [num_patches, hidden]
    pub position_embedding: Vec<f32>,  // [196, 768]

    // Shared ViT backbone
    pub backbone: ViTBackbone,

    // Post-encoder LayerNorm
    pub post_ln_gamma: Vec<f16>,
    pub post_ln_beta: Vec<f16>,

    // MAP pooling head
    pub map_probe: Vec<f32>,           // [768]
    pub map_q_weight: Vec<f32>,        // [768, 768] for probe query
    pub map_q_bias: Vec<f32>,          // [768]
    pub map_k_weight: Vec<f32>,        // [768, 768] for encoder keys
    pub map_k_bias: Vec<f32>,          // [768]
    pub map_v_weight: Vec<f32>,        // [768, 768] for encoder values
    pub map_v_bias: Vec<f32>,          // [768]
    pub map_out_weight: Weight,        // [768, 768]
    pub map_out_bias: Vec<f32>,        // [768]
    pub map_ln_gamma: Vec<f16>,        // [768]
    pub map_ln_beta: Vec<f16>,         // [768]
    pub map_mlp_fc1: Weight,           // [768, 3072]
    pub map_mlp_fc1_bias: Vec<f32>,    // [3072]
    pub map_mlp_fc2: Weight,           // [3072, 768]
    pub map_mlp_fc2_bias: Vec<f32>,    // [768]

    pub num_patches: usize,            // 196
    pub patch_size: usize,             // 16
    pub num_channels: usize,           // 3
}

impl SigLIPVisionWeights {
    pub fn memory_bytes(&self) -> usize {
        self.patch_embed_weight.len() * 4 + self.patch_embed_bias.len() * 4 +
        self.position_embedding.len() * 4 + self.backbone.memory_bytes() +
        (self.post_ln_gamma.len() + self.post_ln_beta.len()) * 2 +
        self.map_probe.len() * 4 +
        (self.map_q_weight.len() + self.map_k_weight.len() + self.map_v_weight.len()) * 4 +
        (self.map_q_bias.len() + self.map_k_bias.len() + self.map_v_bias.len()) * 4 +
        self.map_out_weight.memory_bytes() +
        self.map_mlp_fc1.memory_bytes() + self.map_mlp_fc2.memory_bytes()
    }
}

// ============================================================
// SigLIP Text Weights
// ============================================================

pub struct SigLIPTextWeights {
    pub token_embedding: Weight,       // [256000, 768]
    pub position_embedding: Vec<f32>,  // [64, 768]
    pub backbone: ViTBackbone,
    pub final_ln_gamma: Vec<f16>,
    pub final_ln_beta: Vec<f16>,
    pub head_weight: Weight,           // [768, 768]
    pub head_bias: Vec<f32>,           // [768]
    pub vocab_size: usize,
    pub max_position: usize,
}

impl SigLIPTextWeights {
    pub fn memory_bytes(&self) -> usize {
        self.token_embedding.memory_bytes() + self.position_embedding.len() * 4 +
        self.backbone.memory_bytes() +
        (self.final_ln_gamma.len() + self.final_ln_beta.len()) * 2 +
        self.head_weight.memory_bytes() + self.head_bias.len() * 4
    }
}

// ============================================================
// Vision Forward Pass
// ============================================================

/// Extract non-overlapping patches and project through Conv2d.
/// image: [3, 224, 224] CHW normalized.
/// Returns: [num_patches, hidden] = [196, 768].
pub fn patch_embed(
    image: &[f32],
    weights: &SigLIPVisionWeights,
) -> Vec<f32> {
    let ps = weights.patch_size;
    let nc = weights.num_channels;
    let img_size = 224; // assumes square
    let grid = img_size / ps; // 14
    let num_patches = grid * grid; // 196
    let hidden = weights.backbone.hidden_size;
    let patch_dim = nc * ps * ps; // 768

    let mut patches = vec![0.0f32; num_patches * patch_dim];

    // Extract patches: image[c, py*ps+dy, px*ps+dx]
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

    // GEMM: [num_patches, patch_dim] × weight^T → [num_patches, hidden]
    // Weight is stored as [hidden, patch_dim] in the file but we loaded it transposed
    // Actually, for Conv2d weights [out_ch, in_ch, kH, kW] reshaped as [hidden, patch_dim]
    // We need: output = patches @ weight^T + bias
    // Since weight is [hidden, patch_dim], we do row-by-row dot products
    let mut output = vec![0.0f32; num_patches * hidden];
    for p in 0..num_patches {
        for h in 0..hidden {
            let mut sum = weights.patch_embed_bias[h];
            for d in 0..patch_dim {
                sum += patches[p * patch_dim + d] * weights.patch_embed_weight[h * patch_dim + d];
            }
            output[p * hidden + h] = sum;
        }
    }

    output
}

/// Full SigLIP vision forward: image → 768-dim embedding.
pub fn siglip_vision_forward(
    image: &[f32],
    weights: &SigLIPVisionWeights,
) -> Vec<f32> {
    let hidden = weights.backbone.hidden_size;
    let num_patches = weights.num_patches;

    // 1. Patch embedding
    eprintln!("  Patch embedding...");
    let mut embeddings = patch_embed(image, weights);

    // 2. Add position embeddings
    for i in 0..num_patches * hidden {
        embeddings[i] += weights.position_embedding[i];
    }

    // 3. ViT backbone
    eprintln!("  ViT backbone ({} layers)...", weights.backbone.layers.len());
    let encoder_output = crate::vit::vit_forward(&embeddings, num_patches, &weights.backbone);

    // 4. Post-LayerNorm
    let mut post_normed = vec![0.0f32; num_patches * hidden];
    for t in 0..num_patches {
        let row = &encoder_output[t * hidden..(t + 1) * hidden];
        let n = gemv::layer_norm_f16(row, &weights.post_ln_gamma, &weights.post_ln_beta,
                                      weights.backbone.ln_eps);
        post_normed[t * hidden..(t + 1) * hidden].copy_from_slice(&n);
    }

    // 5. MAP pooling
    eprintln!("  MAP pooling...");
    map_pooling(&post_normed, num_patches, weights)
}

/// MAP (Multi-head Attention Pooling) head.
/// encoder_output: [seq_len, hidden].
/// Returns: [hidden] single embedding vector.
fn map_pooling(
    encoder_output: &[f32],
    seq_len: usize,
    weights: &SigLIPVisionWeights,
) -> Vec<f32> {
    let hidden = weights.backbone.hidden_size;
    let num_heads = weights.backbone.num_heads;
    let head_dim = weights.backbone.head_dim;
    let eps = weights.backbone.ln_eps;

    // Query from probe
    let mut q = vec![0.0f32; hidden];
    for h in 0..hidden {
        let mut sum = weights.map_q_bias[h];
        for d in 0..hidden {
            sum += weights.map_probe[d] * weights.map_q_weight[d * hidden + h];
        }
        q[h] = sum;
    }

    // K, V from encoder output (using f32 GEMM)
    let mut k_all = vec![0.0f32; seq_len * hidden];
    let mut v_all = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        for h in 0..hidden {
            let mut k_sum = weights.map_k_bias[h];
            let mut v_sum = weights.map_v_bias[h];
            for d in 0..hidden {
                let inp = encoder_output[t * hidden + d];
                k_sum += inp * weights.map_k_weight[d * hidden + h];
                v_sum += inp * weights.map_v_weight[d * hidden + h];
            }
            k_all[t * hidden + h] = k_sum;
            v_all[t * hidden + h] = v_sum;
        }
    }

    // Multi-head cross-attention: Q (1 query) attends to K,V (seq_len keys)
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_output = vec![0.0f32; hidden];

    for head in 0..num_heads {
        let ho = head * head_dim;

        // Scores: Q_h dot K_h for each position
        let mut scores = vec![0.0f32; seq_len];
        for ki in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[ho + d] * k_all[ki * hidden + ho + d];
            }
            scores[ki] = dot * scale;
        }

        gemv::softmax_raw(&mut scores);

        // Weighted sum of values
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for ki in 0..seq_len {
                sum += scores[ki] * v_all[ki * hidden + ho + d];
            }
            attn_output[ho + d] = sum;
        }
    }

    // Output projection
    let projected = gemv::gemv_bias(&attn_output, &weights.map_out_weight, &weights.map_out_bias);

    // Residual + LayerNorm + MLP
    let residual = projected;
    let normed = gemv::layer_norm_f16(&residual, &weights.map_ln_gamma, &weights.map_ln_beta, eps);

    // MLP: fc1 + gelu + fc2
    let mut fc1 = gemv::gemv_bias(&normed, &weights.map_mlp_fc1, &weights.map_mlp_fc1_bias);
    for v in fc1.iter_mut() { *v = gemv::gelu_tanh(*v); }
    let fc2 = gemv::gemv_bias(&fc1, &weights.map_mlp_fc2, &weights.map_mlp_fc2_bias);

    // Add residual
    let mut output = vec![0.0f32; hidden];
    for i in 0..hidden {
        output[i] = residual[i] + fc2[i];
    }

    output
}

// ============================================================
// Text Forward Pass
// ============================================================

/// SigLIP text forward: token_ids → 768-dim embedding.
pub fn siglip_text_forward(
    token_ids: &[u32],
    weights: &SigLIPTextWeights,
) -> Vec<f32> {
    let hidden = weights.backbone.hidden_size;
    let seq_len = token_ids.len();

    // 1. Token + position embeddings
    let mut embeddings = vec![0.0f32; seq_len * hidden];
    for (t, &tid) in token_ids.iter().enumerate() {
        let tok_embed = gemv::embed_lookup(&weights.token_embedding, tid as usize, hidden);
        for h in 0..hidden {
            embeddings[t * hidden + h] = tok_embed[h] + weights.position_embedding[t * hidden + h];
        }
    }

    // 2. ViT backbone
    let encoder_output = crate::vit::vit_forward(&embeddings, seq_len, &weights.backbone);

    // 3. Final LayerNorm
    let last_token = &encoder_output[(seq_len - 1) * hidden..seq_len * hidden];
    let normed = gemv::layer_norm_f16(last_token, &weights.final_ln_gamma, &weights.final_ln_beta,
                                       weights.backbone.ln_eps);

    // 4. Head projection
    gemv::gemv_bias(&normed, &weights.head_weight, &weights.head_bias)
}

// ============================================================
// Contrastive Scoring
// ============================================================

/// L2-normalize a vector in place.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() { *x *= inv; }
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// SigLIP contrastive score.
pub fn siglip_score(image_embed: &[f32], text_embed: &[f32], logit_scale: f32, logit_bias: f32) -> f32 {
    let sim = cosine_similarity(image_embed, text_embed);
    gemv::sigmoid(sim * logit_scale.exp() + logit_bias)
}

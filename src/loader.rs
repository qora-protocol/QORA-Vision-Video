//! Safetensors loading for SigLIP 2 and ViViT.
//!
//! Handles: f32, bf16, f16 dtypes. Transposes [out, in] → [in, out] for GEMV.

use half::f16;
use safetensors::SafeTensors;
use std::path::Path;

use crate::gemv::{self, Weight, build_weight};
use crate::vit::{ViTLayerWeights, ViTBackbone};
use crate::siglip::{SigLIPVisionWeights, SigLIPTextWeights};
use crate::vivit::ViViTWeights;

// ============================================================
// Tensor reading helpers
// ============================================================

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    let count = bytes.len() / 2;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        let f32_bits = (bits as u32) << 16;
        out.push(f32::from_bits(f32_bits));
    }
    out
}

fn f16_to_f32(bytes: &[u8]) -> Vec<f32> {
    let count = bytes.len() / 2;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        out.push(f16::from_bits(bits).to_f32());
    }
    out
}

fn f32_from_bytes(bytes: &[u8]) -> Vec<f32> {
    let count = bytes.len() / 4;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let bits = u32::from_le_bytes([
            bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3],
        ]);
        out.push(f32::from_bits(bits));
    }
    out
}

/// Read a tensor as f32, auto-detecting dtype.
fn read_f32(st: &SafeTensors, key: &str) -> Vec<f32> {
    let t = st.tensor(key).unwrap_or_else(|_| panic!("Missing tensor: {key}"));
    match t.dtype() {
        safetensors::Dtype::F32 => f32_from_bytes(t.data()),
        safetensors::Dtype::BF16 => bf16_to_f32(t.data()),
        safetensors::Dtype::F16 => f16_to_f32(t.data()),
        other => panic!("Unsupported dtype {other:?} for {key}"),
    }
}

/// Read a tensor as f16 (for LayerNorm gamma/beta).
fn read_f16(st: &SafeTensors, key: &str) -> Vec<f16> {
    let data = read_f32(st, key);
    gemv::f32_to_f16(&data)
}

/// Transpose [rows, cols] → [cols, rows].
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Read a linear weight, transpose [out, in] → [in, out], and build Weight.
fn read_linear(st: &SafeTensors, key: &str, out_dim: usize, in_dim: usize, use_q4: bool) -> Weight {
    let data = read_f32(st, key);
    let transposed = transpose(&data, out_dim, in_dim);
    build_weight(&transposed, in_dim, out_dim, use_q4)
}

/// Read an embedding weight (no transpose). Stored as [vocab, hidden].
fn read_embedding(st: &SafeTensors, key: &str, vocab: usize, hidden: usize, use_q4: bool) -> Weight {
    let data = read_f32(st, key);
    build_weight(&data, vocab, hidden, use_q4)
}

/// Read a bias vector as f32.
fn read_bias(st: &SafeTensors, key: &str) -> Vec<f32> {
    read_f32(st, key)
}

/// Check if a tensor exists.
#[allow(dead_code)]
fn has_tensor(st: &SafeTensors, key: &str) -> bool {
    st.tensor(key).is_ok()
}

// ============================================================
// Shared ViT layer loading
// ============================================================

fn load_vit_layer(
    st: &SafeTensors,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
    use_q4: bool,
    // Different models use different weight key patterns
    q_key: &str, k_key: &str, v_key: &str, o_key: &str,
    q_bias_key: &str, k_bias_key: &str, v_bias_key: &str, o_bias_key: &str,
    ln1_w_key: &str, ln1_b_key: &str,
    ln2_w_key: &str, ln2_b_key: &str,
    fc1_w_key: &str, fc1_b_key: &str,
    fc2_w_key: &str, fc2_b_key: &str,
) -> ViTLayerWeights {
    ViTLayerWeights {
        q_proj: read_linear(st, &format!("{prefix}{q_key}"), hidden, hidden, use_q4),
        k_proj: read_linear(st, &format!("{prefix}{k_key}"), hidden, hidden, use_q4),
        v_proj: read_linear(st, &format!("{prefix}{v_key}"), hidden, hidden, use_q4),
        o_proj: read_linear(st, &format!("{prefix}{o_key}"), hidden, hidden, use_q4),
        q_bias: read_bias(st, &format!("{prefix}{q_bias_key}")),
        k_bias: read_bias(st, &format!("{prefix}{k_bias_key}")),
        v_bias: read_bias(st, &format!("{prefix}{v_bias_key}")),
        o_bias: read_bias(st, &format!("{prefix}{o_bias_key}")),
        ln1_gamma: read_f16(st, &format!("{prefix}{ln1_w_key}")),
        ln1_beta: read_f16(st, &format!("{prefix}{ln1_b_key}")),
        ln2_gamma: read_f16(st, &format!("{prefix}{ln2_w_key}")),
        ln2_beta: read_f16(st, &format!("{prefix}{ln2_b_key}")),
        mlp_fc1: read_linear(st, &format!("{prefix}{fc1_w_key}"), intermediate, hidden, use_q4),
        mlp_fc1_bias: read_bias(st, &format!("{prefix}{fc1_b_key}")),
        mlp_fc2: read_linear(st, &format!("{prefix}{fc2_w_key}"), hidden, intermediate, use_q4),
        mlp_fc2_bias: read_bias(st, &format!("{prefix}{fc2_b_key}")),
    }
}

// ============================================================
// SigLIP Loading
// ============================================================

/// Load SigLIP 2 model from safetensors.
pub fn load_siglip(
    model_path: &Path,
    use_q4: bool,
) -> Result<(SigLIPVisionWeights, SigLIPTextWeights, f32, f32), Box<dyn std::error::Error>> {
    let st_path = model_path.join("model.safetensors");
    eprintln!("Loading QORA image model from {}...", st_path.display());
    let file_data = std::fs::read(&st_path)?;
    let st = SafeTensors::deserialize(&file_data)?;

    let hidden = 768;
    let intermediate = 3072;
    let num_heads = 12;
    let head_dim = 64;
    let eps = 1e-6f32;

    // === Vision encoder ===
    eprintln!("  Loading vision encoder (12 layers)...");
    let mut vision_layers = Vec::with_capacity(12);
    for i in 0..12 {
        let prefix = format!("vision_model.encoder.layers.{i}.");
        if i % 4 == 0 { eprintln!("    Layer {i}/12..."); }
        vision_layers.push(load_vit_layer(
            &st, &prefix, hidden, intermediate, use_q4,
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.out_proj.weight",
            "self_attn.q_proj.bias", "self_attn.k_proj.bias",
            "self_attn.v_proj.bias", "self_attn.out_proj.bias",
            "layer_norm1.weight", "layer_norm1.bias",
            "layer_norm2.weight", "layer_norm2.bias",
            "mlp.fc1.weight", "mlp.fc1.bias",
            "mlp.fc2.weight", "mlp.fc2.bias",
        ));
    }

    // Patch embedding
    let patch_embed_weight = read_f32(&st, "vision_model.embeddings.patch_embedding.weight");
    let patch_embed_bias = read_bias(&st, "vision_model.embeddings.patch_embedding.bias");

    // Position embedding
    let position_embedding = read_f32(&st, "vision_model.embeddings.position_embedding.weight");

    // Post-LayerNorm
    let post_ln_gamma = read_f16(&st, "vision_model.post_layernorm.weight");
    let post_ln_beta = read_f16(&st, "vision_model.post_layernorm.bias");

    // MAP head
    eprintln!("  Loading MAP pooling head...");
    let map_probe = {
        let raw = read_f32(&st, "vision_model.head.probe");
        // probe is [1, 1, 768], flatten to [768]
        raw
    };

    // in_proj_weight is packed [2304, 768] = [Q;K;V]
    let in_proj_w = read_f32(&st, "vision_model.head.attention.in_proj_weight");
    let in_proj_b = read_f32(&st, "vision_model.head.attention.in_proj_bias");

    // Split Q, K, V weights (each [768, 768]) and transpose for GEMV [in, out]
    let map_q_weight = transpose(&in_proj_w[0..768 * 768], 768, 768);
    let map_k_weight = transpose(&in_proj_w[768 * 768..2 * 768 * 768], 768, 768);
    let map_v_weight = transpose(&in_proj_w[2 * 768 * 768..3 * 768 * 768], 768, 768);
    let map_q_bias = in_proj_b[0..768].to_vec();
    let map_k_bias = in_proj_b[768..2 * 768].to_vec();
    let map_v_bias = in_proj_b[2 * 768..3 * 768].to_vec();

    let map_out_weight = read_linear(&st, "vision_model.head.attention.out_proj.weight", hidden, hidden, use_q4);
    let map_out_bias = read_bias(&st, "vision_model.head.attention.out_proj.bias");
    let map_ln_gamma = read_f16(&st, "vision_model.head.layernorm.weight");
    let map_ln_beta = read_f16(&st, "vision_model.head.layernorm.bias");
    let map_mlp_fc1 = read_linear(&st, "vision_model.head.mlp.fc1.weight", intermediate, hidden, use_q4);
    let map_mlp_fc1_bias = read_bias(&st, "vision_model.head.mlp.fc1.bias");
    let map_mlp_fc2 = read_linear(&st, "vision_model.head.mlp.fc2.weight", hidden, intermediate, use_q4);
    let map_mlp_fc2_bias = read_bias(&st, "vision_model.head.mlp.fc2.bias");

    let vision = SigLIPVisionWeights {
        patch_embed_weight,
        patch_embed_bias,
        position_embedding,
        backbone: ViTBackbone {
            layers: vision_layers,
            hidden_size: hidden,
            num_heads,
            head_dim,
            intermediate_size: intermediate,
            ln_eps: eps,
        },
        post_ln_gamma,
        post_ln_beta,
        map_probe,
        map_q_weight,
        map_q_bias,
        map_k_weight,
        map_k_bias,
        map_v_weight,
        map_v_bias,
        map_out_weight,
        map_out_bias,
        map_ln_gamma,
        map_ln_beta,
        map_mlp_fc1,
        map_mlp_fc1_bias,
        map_mlp_fc2,
        map_mlp_fc2_bias,
        num_patches: 196,
        patch_size: 16,
        num_channels: 3,
    };

    // === Text encoder ===
    eprintln!("  Loading text encoder (12 layers)...");
    let mut text_layers = Vec::with_capacity(12);
    for i in 0..12 {
        let prefix = format!("text_model.encoder.layers.{i}.");
        if i % 4 == 0 { eprintln!("    Layer {i}/12..."); }
        text_layers.push(load_vit_layer(
            &st, &prefix, hidden, intermediate, use_q4,
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.out_proj.weight",
            "self_attn.q_proj.bias", "self_attn.k_proj.bias",
            "self_attn.v_proj.bias", "self_attn.out_proj.bias",
            "layer_norm1.weight", "layer_norm1.bias",
            "layer_norm2.weight", "layer_norm2.bias",
            "mlp.fc1.weight", "mlp.fc1.bias",
            "mlp.fc2.weight", "mlp.fc2.bias",
        ));
    }

    // Token embedding [256000, 768] — no transpose, it's already [vocab, hidden]
    let token_embedding = read_embedding(&st, "text_model.embeddings.token_embedding.weight", 256000, hidden, use_q4);
    // Position embedding [64, 768]
    let text_pos_embed = read_f32(&st, "text_model.embeddings.position_embedding.weight");

    let text_final_ln_gamma = read_f16(&st, "text_model.final_layer_norm.weight");
    let text_final_ln_beta = read_f16(&st, "text_model.final_layer_norm.bias");

    let text_head_weight = read_linear(&st, "text_model.head.weight", hidden, hidden, use_q4);
    let text_head_bias = read_bias(&st, "text_model.head.bias");

    let text = SigLIPTextWeights {
        token_embedding,
        position_embedding: text_pos_embed,
        backbone: ViTBackbone {
            layers: text_layers,
            hidden_size: hidden,
            num_heads,
            head_dim,
            intermediate_size: intermediate,
            ln_eps: eps,
        },
        final_ln_gamma: text_final_ln_gamma,
        final_ln_beta: text_final_ln_beta,
        head_weight: text_head_weight,
        head_bias: text_head_bias,
        vocab_size: 256000,
        max_position: 64,
    };

    // Logit scale and bias
    let logit_scale = read_f32(&st, "logit_scale")[0];
    let logit_bias = read_f32(&st, "logit_bias")[0];

    let vmb = vision.memory_bytes() / (1024 * 1024);
    let tmb = text.memory_bytes() / (1024 * 1024);
    eprintln!("  Vision: {vmb} MB, Text: {tmb} MB");

    Ok((vision, text, logit_scale, logit_bias))
}

// ============================================================
// ViViT Loading
// ============================================================

/// Load ViViT from safetensors (converted from pytorch_model.bin).
pub fn load_vivit(
    model_path: &Path,
    use_q4: bool,
) -> Result<ViViTWeights, Box<dyn std::error::Error>> {
    let st_path = model_path.join("model.safetensors");
    eprintln!("Loading QORA video model from {}...", st_path.display());
    let file_data = std::fs::read(&st_path)?;
    let st = SafeTensors::deserialize(&file_data)?;

    let hidden = 768;
    let intermediate = 3072;
    let num_heads = 12;
    let head_dim = 64;
    let eps = 1e-6f32;

    // Tubelet embedding
    eprintln!("  Loading tubelet embedding...");
    let tubelet_weight = read_f32(&st, "vivit.embeddings.patch_embeddings.projection.weight");
    let tubelet_bias = read_bias(&st, "vivit.embeddings.patch_embeddings.projection.bias");

    // CLS token [1, 1, 768] → [768]
    let cls_token = read_f32(&st, "vivit.embeddings.cls_token");

    // Position embeddings [1, 3137, 768] → [3137, 768]
    let position_embedding = read_f32(&st, "vivit.embeddings.position_embeddings");

    // Transformer layers
    eprintln!("  Loading transformer (12 layers)...");
    let mut layers = Vec::with_capacity(12);
    for i in 0..12 {
        let prefix = format!("vivit.encoder.layer.{i}.");
        if i % 4 == 0 { eprintln!("    Layer {i}/12..."); }
        layers.push(load_vit_layer(
            &st, &prefix, hidden, intermediate, use_q4,
            "attention.attention.query.weight", "attention.attention.key.weight",
            "attention.attention.value.weight", "attention.output.dense.weight",
            "attention.attention.query.bias", "attention.attention.key.bias",
            "attention.attention.value.bias", "attention.output.dense.bias",
            "layernorm_before.weight", "layernorm_before.bias",
            "layernorm_after.weight", "layernorm_after.bias",
            "intermediate.dense.weight", "intermediate.dense.bias",
            "output.dense.weight", "output.dense.bias",
        ));
    }

    // Final LayerNorm
    let final_ln_gamma = read_f16(&st, "vivit.layernorm.weight");
    let final_ln_beta = read_f16(&st, "vivit.layernorm.bias");

    // Classifier
    let classifier_weight = read_linear(&st, "classifier.weight", 400, hidden, use_q4);
    let classifier_bias = read_bias(&st, "classifier.bias");

    let weights = ViViTWeights {
        tubelet_weight,
        tubelet_bias,
        cls_token,
        position_embedding,
        backbone: ViTBackbone {
            layers,
            hidden_size: hidden,
            num_heads,
            head_dim,
            intermediate_size: intermediate,
            ln_eps: eps,
        },
        final_ln_gamma,
        final_ln_beta,
        classifier_weight,
        classifier_bias,
        num_patches: 3136,
        num_frames: 32,
        tubelet_size: [2, 16, 16],
    };

    let mb = weights.memory_bytes() / (1024 * 1024);
    eprintln!("  Video model loaded: {mb} MB");

    Ok(weights)
}

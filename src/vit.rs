//! Shared ViT (Vision Transformer) backbone.
//!
//! Both SigLIP 2 and ViViT use identical ViT-Base transformer layers:
//!   Pre-norm attention + Pre-norm MLP, bidirectional (no causal mask), no KV cache.

use half::f16;
use crate::gemv::{self, Weight};

// ============================================================
// Weight structures
// ============================================================

pub struct ViTLayerWeights {
    // Self-attention projections
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub o_bias: Vec<f32>,

    // LayerNorm 1 (pre-attention)
    pub ln1_gamma: Vec<f16>,
    pub ln1_beta: Vec<f16>,

    // MLP
    pub mlp_fc1: Weight,
    pub mlp_fc1_bias: Vec<f32>,
    pub mlp_fc2: Weight,
    pub mlp_fc2_bias: Vec<f32>,

    // LayerNorm 2 (pre-MLP)
    pub ln2_gamma: Vec<f16>,
    pub ln2_beta: Vec<f16>,
}

pub struct ViTBackbone {
    pub layers: Vec<ViTLayerWeights>,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub ln_eps: f32,
}

impl ViTBackbone {
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| {
            l.q_proj.memory_bytes() + l.k_proj.memory_bytes() +
            l.v_proj.memory_bytes() + l.o_proj.memory_bytes() +
            l.mlp_fc1.memory_bytes() + l.mlp_fc2.memory_bytes() +
            (l.q_bias.len() + l.k_bias.len() + l.v_bias.len() + l.o_bias.len() +
             l.mlp_fc1_bias.len() + l.mlp_fc2_bias.len()) * 4 +
            (l.ln1_gamma.len() + l.ln1_beta.len() +
             l.ln2_gamma.len() + l.ln2_beta.len()) * 2
        }).sum()
    }
}

// ============================================================
// Forward pass
// ============================================================

/// Forward pass through the full ViT backbone.
/// x: [seq_len * hidden_size] flattened, row-major.
/// Returns: [seq_len * hidden_size].
pub fn vit_forward(x: &[f32], seq_len: usize, backbone: &ViTBackbone) -> Vec<f32> {
    let _hidden = backbone.hidden_size;
    let mut h = x.to_vec();

    for (i, layer) in backbone.layers.iter().enumerate() {
        if i % 4 == 0 {
            eprintln!("    ViT layer {i}/{}", backbone.layers.len());
        }
        h = vit_layer_forward(&h, seq_len, layer, backbone);
    }

    h
}

/// Forward pass through one ViT transformer layer.
fn vit_layer_forward(
    x: &[f32],
    seq_len: usize,
    layer: &ViTLayerWeights,
    config: &ViTBackbone,
) -> Vec<f32> {
    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let eps = config.ln_eps;

    // --- Pre-norm + Self-Attention ---
    // Apply LayerNorm 1
    let mut normed = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = &x[t * hidden..(t + 1) * hidden];
        let n = gemv::layer_norm_f16(row, &layer.ln1_gamma, &layer.ln1_beta, eps);
        normed[t * hidden..(t + 1) * hidden].copy_from_slice(&n);
    }

    // Q, K, V projections with bias
    let q = gemv::gemm_bias(&normed, seq_len, &layer.q_proj, &layer.q_bias);
    let k = gemv::gemm_bias(&normed, seq_len, &layer.k_proj, &layer.k_bias);
    let v = gemv::gemm_bias(&normed, seq_len, &layer.v_proj, &layer.v_bias);

    // Multi-head attention (process one head at a time for memory efficiency)
    let mut attn_output = vec![0.0f32; seq_len * hidden];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        // Extract Q, K, V for this head
        let mut scores = vec![0.0f32; seq_len * seq_len];

        // Compute attention scores: Q_h @ K_h^T * scale
        for qi in 0..seq_len {
            for ki in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * hidden + h * head_dim + d] *
                           k[ki * hidden + h * head_dim + d];
                }
                scores[qi * seq_len + ki] = dot * scale;
            }
        }

        // Softmax per query (no causal mask — bidirectional)
        for qi in 0..seq_len {
            gemv::softmax_raw(&mut scores[qi * seq_len..(qi + 1) * seq_len]);
        }

        // Weighted sum of values
        for qi in 0..seq_len {
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..seq_len {
                    sum += scores[qi * seq_len + ki] *
                           v[ki * hidden + h * head_dim + d];
                }
                attn_output[qi * hidden + h * head_dim + d] = sum;
            }
        }
    }

    // Output projection + residual
    let attn_proj = gemv::gemm_bias(&attn_output, seq_len, &layer.o_proj, &layer.o_bias);
    let mut h = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden {
        h[i] = x[i] + attn_proj[i];
    }

    // --- Pre-norm + MLP ---
    let mut normed2 = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = &h[t * hidden..(t + 1) * hidden];
        let n = gemv::layer_norm_f16(row, &layer.ln2_gamma, &layer.ln2_beta, eps);
        normed2[t * hidden..(t + 1) * hidden].copy_from_slice(&n);
    }

    // fc1 + gelu_tanh + fc2
    let mut fc1_out = gemv::gemm_bias(&normed2, seq_len, &layer.mlp_fc1, &layer.mlp_fc1_bias);
    for v in fc1_out.iter_mut() {
        *v = gemv::gelu_tanh(*v);
    }
    let fc2_out = gemv::gemm_bias(&fc1_out, seq_len, &layer.mlp_fc2, &layer.mlp_fc2_bias);

    // Residual
    for i in 0..seq_len * hidden {
        h[i] += fc2_out[i];
    }

    h
}

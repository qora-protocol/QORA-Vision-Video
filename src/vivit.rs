//! ViViT (Video Vision Transformer) — video → action classification.
//!
//! Pipeline: Conv3d tubelet embed → CLS + position → ViT → LN → classifier
//! 32 frames × 224×224 → 3136 tubelets → 3137 tokens → 400-class logits

use half::f16;
use crate::gemv;
use crate::vit::ViTBackbone;

// ============================================================
// Weight structures
// ============================================================

pub struct ViViTWeights {
    // Tubelet embedding: Conv3d(3, 768, k=[2,16,16], s=[2,16,16])
    pub tubelet_weight: Vec<f32>,       // [768, 1536] (1536 = 3*2*16*16)
    pub tubelet_bias: Vec<f32>,         // [768]

    // CLS token
    pub cls_token: Vec<f32>,            // [768]

    // Position embeddings
    pub position_embedding: Vec<f32>,   // [seq_len, 768] (seq_len = 1 + num_patches)

    // Shared ViT backbone
    pub backbone: ViTBackbone,

    // Final LayerNorm
    pub final_ln_gamma: Vec<f16>,
    pub final_ln_beta: Vec<f16>,

    // Classifier head
    pub classifier_weight: gemv::Weight,  // [768, 400]
    pub classifier_bias: Vec<f32>,        // [400]

    pub num_patches: usize,             // 3136
    pub num_frames: usize,              // 32
    pub tubelet_size: [usize; 3],       // [2, 16, 16]
}

impl ViViTWeights {
    pub fn memory_bytes(&self) -> usize {
        self.tubelet_weight.len() * 4 + self.tubelet_bias.len() * 4 +
        self.cls_token.len() * 4 + self.position_embedding.len() * 4 +
        self.backbone.memory_bytes() +
        (self.final_ln_gamma.len() + self.final_ln_beta.len()) * 2 +
        self.classifier_weight.memory_bytes() + self.classifier_bias.len() * 4
    }

    pub fn seq_len(&self) -> usize { 1 + self.num_patches }
}

// ============================================================
// Tubelet Embedding (3D Conv)
// ============================================================

/// Extract 3D tubelets from video and project.
/// video: [3, num_frames, 224, 224] in CTHW layout.
/// Returns: [num_patches, hidden] where num_patches = (T/2) * (H/16) * (W/16) = 3136.
fn tubelet_embed(
    video: &[f32],
    weights: &ViViTWeights,
) -> Vec<f32> {
    let [tt, th, tw] = weights.tubelet_size;
    let nc = 3usize;
    let nf = weights.num_frames;
    let img_size = 224usize;
    let hidden = weights.backbone.hidden_size;

    let grid_t = nf / tt;          // 16
    let grid_h = img_size / th;    // 14
    let grid_w = img_size / tw;    // 14
    let num_patches = grid_t * grid_h * grid_w; // 3136
    let tubelet_dim = nc * tt * th * tw; // 1536

    let mut tubelets = vec![0.0f32; num_patches * tubelet_dim];

    // Extract tubelets: video[c, pt*tt+dt, py*th+dy, px*tw+dx]
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
                                // video layout: [C, T, H, W]
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

    // GEMM: [num_patches, tubelet_dim] × weight^T → [num_patches, hidden]
    // Weight stored as [hidden, tubelet_dim]
    let mut output = vec![0.0f32; num_patches * hidden];
    for p in 0..num_patches {
        for h in 0..hidden {
            let mut sum = weights.tubelet_bias[h];
            for d in 0..tubelet_dim {
                sum += tubelets[p * tubelet_dim + d] * weights.tubelet_weight[h * tubelet_dim + d];
            }
            output[p * hidden + h] = sum;
        }
    }

    output
}

// ============================================================
// Full Forward Pass
// ============================================================

/// Full ViViT forward: video → embedding or classification logits.
/// video: [3, num_frames, 224, 224] in CTHW layout, normalized.
/// Returns: (embedding, logits) where embedding is CLS [768] and logits is [400].
pub fn vivit_forward(
    video: &[f32],
    weights: &ViViTWeights,
) -> (Vec<f32>, Vec<f32>) {
    let hidden = weights.backbone.hidden_size;
    let num_patches = weights.num_patches;
    let seq_len = 1 + num_patches; // CLS + patches

    // 1. Tubelet embedding
    eprintln!("  Tubelet embedding ({num_patches} patches)...");
    let patch_embeds = tubelet_embed(video, weights);

    // 2. Prepend CLS + add position embeddings
    let mut embeddings = vec![0.0f32; seq_len * hidden];

    // CLS token at position 0
    for h in 0..hidden {
        embeddings[h] = weights.cls_token[h] + weights.position_embedding[h];
    }

    // Patch tokens at positions 1..3137
    for p in 0..num_patches {
        for h in 0..hidden {
            embeddings[(1 + p) * hidden + h] =
                patch_embeds[p * hidden + h] + weights.position_embedding[(1 + p) * hidden + h];
        }
    }

    // 3. ViT backbone
    eprintln!("  ViT backbone ({} layers, seq_len={seq_len})...", weights.backbone.layers.len());
    let encoder_output = crate::vit::vit_forward(&embeddings, seq_len, &weights.backbone);

    // 4. Final LayerNorm on CLS token
    let cls_hidden = &encoder_output[0..hidden];
    let cls_normed = gemv::layer_norm_f16(cls_hidden, &weights.final_ln_gamma,
                                           &weights.final_ln_beta, weights.backbone.ln_eps);

    // 5. Classifier
    let logits = gemv::gemv_bias(&cls_normed, &weights.classifier_weight, &weights.classifier_bias);

    (cls_normed, logits)
}

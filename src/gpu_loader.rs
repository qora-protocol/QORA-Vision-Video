//! GPU weight loading for QORA-Vision.
//!
//! Converts CPU weights (Q4/F16) into Cortex GPU tensors.
//! Reuses same Q4 conversion as QOR3B: XOR 0x88 nibble flip, TensorData::from_bytes_vec.

use cortex::prelude::*;
use cortex::tensor::{DType, TensorData};
use cortex::tensor::quantization::{
    BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use half::f16;

use crate::gemv::Weight;
use crate::siglip::{SigLIPVisionWeights, SigLIPTextWeights};
use crate::vivit::ViViTWeights;
use crate::vit::ViTLayerWeights;

// ============================================================
// GPU model structures
// ============================================================

pub struct GpuViTLayer<B: Backend> {
    pub q_proj: Tensor<B, 2>,
    pub k_proj: Tensor<B, 2>,
    pub v_proj: Tensor<B, 2>,
    pub o_proj: Tensor<B, 2>,
    pub q_bias: Tensor<B, 1>,
    pub k_bias: Tensor<B, 1>,
    pub v_bias: Tensor<B, 1>,
    pub o_bias: Tensor<B, 1>,
    pub ln1_gamma: Tensor<B, 1>,
    pub ln1_beta: Tensor<B, 1>,
    pub ln2_gamma: Tensor<B, 1>,
    pub ln2_beta: Tensor<B, 1>,
    pub mlp_fc1: Tensor<B, 2>,
    pub mlp_fc2: Tensor<B, 2>,
    pub mlp_fc1_bias: Tensor<B, 1>,
    pub mlp_fc2_bias: Tensor<B, 1>,
}

pub struct GpuSigLIPVision<B: Backend> {
    pub patch_embed_weight: Tensor<B, 2>,
    pub patch_embed_bias: Tensor<B, 1>,
    pub position_embedding: Tensor<B, 2>,
    pub layers: Vec<GpuViTLayer<B>>,
    pub post_ln_gamma: Tensor<B, 1>,
    pub post_ln_beta: Tensor<B, 1>,
    // MAP pooling
    pub map_probe: Tensor<B, 1>,
    pub map_q_weight: Tensor<B, 2>,
    pub map_q_bias: Tensor<B, 1>,
    pub map_k_weight: Tensor<B, 2>,
    pub map_k_bias: Tensor<B, 1>,
    pub map_v_weight: Tensor<B, 2>,
    pub map_v_bias: Tensor<B, 1>,
    pub map_out_weight: Tensor<B, 2>,
    pub map_out_bias: Tensor<B, 1>,
    pub map_ln_gamma: Tensor<B, 1>,
    pub map_ln_beta: Tensor<B, 1>,
    pub map_mlp_fc1: Tensor<B, 2>,
    pub map_mlp_fc1_bias: Tensor<B, 1>,
    pub map_mlp_fc2: Tensor<B, 2>,
    pub map_mlp_fc2_bias: Tensor<B, 1>,
    pub num_patches: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

pub struct GpuSigLIPText<B: Backend> {
    // token_embedding kept on CPU (Q4 select not supported on GPU)
    pub position_embedding: Tensor<B, 2>,
    pub layers: Vec<GpuViTLayer<B>>,
    pub final_ln_gamma: Tensor<B, 1>,
    pub final_ln_beta: Tensor<B, 1>,
    pub head_weight: Tensor<B, 2>,
    pub head_bias: Tensor<B, 1>,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

pub struct GpuViViT<B: Backend> {
    pub tubelet_weight: Tensor<B, 2>,
    pub tubelet_bias: Tensor<B, 1>,
    pub cls_token: Tensor<B, 1>,
    pub position_embedding: Tensor<B, 2>,
    pub layers: Vec<GpuViTLayer<B>>,
    pub final_ln_gamma: Tensor<B, 1>,
    pub final_ln_beta: Tensor<B, 1>,
    pub classifier_weight: Tensor<B, 2>,
    pub classifier_bias: Tensor<B, 1>,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_patches: usize,
}

// ============================================================
// Q4 format conversion (same as QOR3B)
// ============================================================

fn q4_scheme() -> QuantScheme {
    QuantScheme {
        value: QuantValue::Q4S,
        param: QuantParam::F32,
        store: QuantStore::PackedU32(0),
        level: QuantLevel::Block(BlockSize::new([32])),
        mode: QuantMode::Symmetric,
    }
}

fn convert_q4_packed(packed: &[u8]) -> Vec<u8> {
    packed.iter().map(|&b| b ^ 0x88).collect()
}

fn q4_weight_to_gpu<B: Backend>(
    packed: &[u8],
    scales: &[f16],
    k: usize,
    n: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let burn_packed = convert_q4_packed(packed);
    let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();
    let scale_bytes: Vec<u8> = scales_f32.iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();

    let mut combined = Vec::with_capacity(burn_packed.len() + scale_bytes.len());
    combined.extend_from_slice(&burn_packed);
    combined.extend_from_slice(&scale_bytes);

    let scheme = q4_scheme();
    let data = TensorData::from_bytes_vec(combined, vec![k, n], DType::QFloat(scheme));
    Tensor::from_data(data, device)
}

fn f16_weight_to_gpu<B: Backend>(
    data: &[f16],
    k: usize,
    n: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    let td = TensorData::new(f32_data, [k, n]);
    Tensor::from_data(td, device)
}

fn weight_to_gpu<B: Backend>(w: &Weight, device: &B::Device) -> Tensor<B, 2> {
    match w {
        Weight::F16(fw) => f16_weight_to_gpu::<B>(&fw.data, fw.k, fw.n, device),
        Weight::Q4(qw) => q4_weight_to_gpu::<B>(&qw.packed, &qw.scales, qw.k, qw.n, device),
    }
}

fn f16_to_gpu_1d<B: Backend>(data: &[f16], device: &B::Device) -> Tensor<B, 1> {
    let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    let td = TensorData::new(f32_data, [data.len()]);
    Tensor::from_data(td, device)
}

fn f32_to_gpu_1d<B: Backend>(data: &[f32], device: &B::Device) -> Tensor<B, 1> {
    let td = TensorData::new(data.to_vec(), [data.len()]);
    Tensor::from_data(td, device)
}

fn f32_to_gpu_2d<B: Backend>(data: &[f32], rows: usize, cols: usize, device: &B::Device) -> Tensor<B, 2> {
    let td = TensorData::new(data.to_vec(), [rows, cols]);
    Tensor::from_data(td, device)
}

// ============================================================
// Load ViT layer
// ============================================================

fn load_vit_layer_gpu<B: Backend>(layer: &ViTLayerWeights, device: &B::Device) -> GpuViTLayer<B> {
    GpuViTLayer {
        q_proj: weight_to_gpu::<B>(&layer.q_proj, device),
        k_proj: weight_to_gpu::<B>(&layer.k_proj, device),
        v_proj: weight_to_gpu::<B>(&layer.v_proj, device),
        o_proj: weight_to_gpu::<B>(&layer.o_proj, device),
        q_bias: f32_to_gpu_1d::<B>(&layer.q_bias, device),
        k_bias: f32_to_gpu_1d::<B>(&layer.k_bias, device),
        v_bias: f32_to_gpu_1d::<B>(&layer.v_bias, device),
        o_bias: f32_to_gpu_1d::<B>(&layer.o_bias, device),
        ln1_gamma: f16_to_gpu_1d::<B>(&layer.ln1_gamma, device),
        ln1_beta: f16_to_gpu_1d::<B>(&layer.ln1_beta, device),
        ln2_gamma: f16_to_gpu_1d::<B>(&layer.ln2_gamma, device),
        ln2_beta: f16_to_gpu_1d::<B>(&layer.ln2_beta, device),
        mlp_fc1: weight_to_gpu::<B>(&layer.mlp_fc1, device),
        mlp_fc2: weight_to_gpu::<B>(&layer.mlp_fc2, device),
        mlp_fc1_bias: f32_to_gpu_1d::<B>(&layer.mlp_fc1_bias, device),
        mlp_fc2_bias: f32_to_gpu_1d::<B>(&layer.mlp_fc2_bias, device),
    }
}

// ============================================================
// Load SigLIP Vision
// ============================================================

pub fn load_siglip_vision_gpu<B: Backend>(
    weights: &SigLIPVisionWeights,
    device: &B::Device,
) -> GpuSigLIPVision<B> {
    let hidden = weights.backbone.hidden_size;
    let num_patches = weights.num_patches;

    eprintln!("Loading SigLIP vision to GPU ({} layers)...", weights.backbone.layers.len());

    let mut layers = Vec::with_capacity(weights.backbone.layers.len());
    for (i, layer) in weights.backbone.layers.iter().enumerate() {
        if i % 4 == 0 {
            eprintln!("  Layer {i}/{}...", weights.backbone.layers.len());
        }
        layers.push(load_vit_layer_gpu::<B>(layer, device));
    }
    eprintln!("  All layers loaded");

    let patch_dim = weights.num_channels * weights.patch_size * weights.patch_size;

    GpuSigLIPVision {
        patch_embed_weight: f32_to_gpu_2d::<B>(&weights.patch_embed_weight, hidden, patch_dim, device),
        patch_embed_bias: f32_to_gpu_1d::<B>(&weights.patch_embed_bias, device),
        position_embedding: f32_to_gpu_2d::<B>(&weights.position_embedding, num_patches, hidden, device),
        layers,
        post_ln_gamma: f16_to_gpu_1d::<B>(&weights.post_ln_gamma, device),
        post_ln_beta: f16_to_gpu_1d::<B>(&weights.post_ln_beta, device),
        map_probe: f32_to_gpu_1d::<B>(&weights.map_probe, device),
        map_q_weight: f32_to_gpu_2d::<B>(&weights.map_q_weight, hidden, hidden, device),
        map_q_bias: f32_to_gpu_1d::<B>(&weights.map_q_bias, device),
        map_k_weight: f32_to_gpu_2d::<B>(&weights.map_k_weight, hidden, hidden, device),
        map_k_bias: f32_to_gpu_1d::<B>(&weights.map_k_bias, device),
        map_v_weight: f32_to_gpu_2d::<B>(&weights.map_v_weight, hidden, hidden, device),
        map_v_bias: f32_to_gpu_1d::<B>(&weights.map_v_bias, device),
        map_out_weight: weight_to_gpu::<B>(&weights.map_out_weight, device),
        map_out_bias: f32_to_gpu_1d::<B>(&weights.map_out_bias, device),
        map_ln_gamma: f16_to_gpu_1d::<B>(&weights.map_ln_gamma, device),
        map_ln_beta: f16_to_gpu_1d::<B>(&weights.map_ln_beta, device),
        map_mlp_fc1: weight_to_gpu::<B>(&weights.map_mlp_fc1, device),
        map_mlp_fc1_bias: f32_to_gpu_1d::<B>(&weights.map_mlp_fc1_bias, device),
        map_mlp_fc2: weight_to_gpu::<B>(&weights.map_mlp_fc2, device),
        map_mlp_fc2_bias: f32_to_gpu_1d::<B>(&weights.map_mlp_fc2_bias, device),
        num_patches,
        hidden_size: hidden,
        num_heads: weights.backbone.num_heads,
        head_dim: weights.backbone.head_dim,
    }
}

// ============================================================
// Load SigLIP Text
// ============================================================

pub fn load_siglip_text_gpu<B: Backend>(
    weights: &SigLIPTextWeights,
    device: &B::Device,
) -> GpuSigLIPText<B> {
    let hidden = weights.backbone.hidden_size;

    eprintln!("Loading SigLIP text to GPU ({} layers)...", weights.backbone.layers.len());

    let mut layers = Vec::with_capacity(weights.backbone.layers.len());
    for (i, layer) in weights.backbone.layers.iter().enumerate() {
        if i % 4 == 0 {
            eprintln!("  Layer {i}/{}...", weights.backbone.layers.len());
        }
        layers.push(load_vit_layer_gpu::<B>(layer, device));
    }
    eprintln!("  All layers loaded");

    GpuSigLIPText {
        // token_embedding stays on CPU — Q4 select not supported on GPU
        position_embedding: f32_to_gpu_2d::<B>(&weights.position_embedding, weights.max_position, hidden, device),
        layers,
        final_ln_gamma: f16_to_gpu_1d::<B>(&weights.final_ln_gamma, device),
        final_ln_beta: f16_to_gpu_1d::<B>(&weights.final_ln_beta, device),
        head_weight: weight_to_gpu::<B>(&weights.head_weight, device),
        head_bias: f32_to_gpu_1d::<B>(&weights.head_bias, device),
        hidden_size: hidden,
        num_heads: weights.backbone.num_heads,
        head_dim: weights.backbone.head_dim,
    }
}

// ============================================================
// Load ViViT
// ============================================================

pub fn load_vivit_gpu<B: Backend>(
    weights: &ViViTWeights,
    device: &B::Device,
) -> GpuViViT<B> {
    let hidden = weights.backbone.hidden_size;
    let seq_len = weights.seq_len();
    let [tt, th, tw] = weights.tubelet_size;
    let tubelet_dim = 3 * tt * th * tw;

    eprintln!("Loading ViViT to GPU ({} layers, seq_len={})...",
        weights.backbone.layers.len(), seq_len);

    let mut layers = Vec::with_capacity(weights.backbone.layers.len());
    for (i, layer) in weights.backbone.layers.iter().enumerate() {
        if i % 4 == 0 {
            eprintln!("  Layer {i}/{}...", weights.backbone.layers.len());
        }
        layers.push(load_vit_layer_gpu::<B>(layer, device));
    }
    eprintln!("  All layers loaded");

    GpuViViT {
        tubelet_weight: f32_to_gpu_2d::<B>(&weights.tubelet_weight, hidden, tubelet_dim, device),
        tubelet_bias: f32_to_gpu_1d::<B>(&weights.tubelet_bias, device),
        cls_token: f32_to_gpu_1d::<B>(&weights.cls_token, device),
        position_embedding: f32_to_gpu_2d::<B>(&weights.position_embedding, seq_len, hidden, device),
        layers,
        final_ln_gamma: f16_to_gpu_1d::<B>(&weights.final_ln_gamma, device),
        final_ln_beta: f16_to_gpu_1d::<B>(&weights.final_ln_beta, device),
        classifier_weight: weight_to_gpu::<B>(&weights.classifier_weight, device),
        classifier_bias: f32_to_gpu_1d::<B>(&weights.classifier_bias, device),
        hidden_size: hidden,
        num_heads: weights.backbone.num_heads,
        head_dim: weights.backbone.head_dim,
        num_patches: weights.num_patches,
    }
}

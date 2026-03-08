//! Configuration structs for SigLIP 2 and ViViT models.

use serde::Deserialize;
use std::path::Path;

// ============================================================
// SigLIP 2 Config
// ============================================================

#[derive(Debug, Deserialize)]
pub struct SigLIPConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_3")]
    pub num_channels: usize,
    #[serde(default = "default_ln_eps")]
    pub layer_norm_eps: f64,
}

fn default_hidden_size() -> usize { 768 }
fn default_num_layers() -> usize { 12 }
fn default_num_heads() -> usize { 12 }
fn default_intermediate() -> usize { 3072 }
fn default_image_size() -> usize { 224 }
fn default_patch_size() -> usize { 16 }
fn default_3() -> usize { 3 }
fn default_ln_eps() -> f64 { 1e-6 }

impl SigLIPConfig {
    pub fn head_dim(&self) -> usize { self.hidden_size / self.num_attention_heads }
    pub fn num_patches(&self) -> usize { (self.image_size / self.patch_size).pow(2) }
}

impl Default for SigLIPConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            layer_norm_eps: 1e-6,
        }
    }
}

// Full SigLIP model config (wraps vision + text)
#[derive(Debug, Deserialize)]
pub struct SigLIPModelConfig {
    #[serde(default)]
    pub vision_config: SigLIPVisionConfigWrapper,
    #[serde(default)]
    pub text_config: SigLIPTextConfigWrapper,
}

#[derive(Debug, Deserialize, Default)]
pub struct SigLIPVisionConfigWrapper {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_ln_eps")]
    pub layer_norm_eps: f64,
}

#[derive(Debug, Deserialize, Default)]
pub struct SigLIPTextConfigWrapper {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_text_vocab")]
    pub vocab_size: usize,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_ln_eps")]
    pub layer_norm_eps: f64,
}

fn default_text_vocab() -> usize { 256000 }
fn default_max_pos() -> usize { 64 }

// ============================================================
// ViViT Config
// ============================================================

#[derive(Debug, Deserialize)]
pub struct ViViTConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_32")]
    pub num_frames: usize,
    #[serde(default = "default_tubelet")]
    pub tubelet_size: Vec<usize>,
    #[serde(default = "default_400")]
    pub num_labels: usize,
    #[serde(default = "default_ln_eps")]
    pub layer_norm_eps: f64,
}

fn default_32() -> usize { 32 }
fn default_tubelet() -> Vec<usize> { vec![2, 16, 16] }
fn default_400() -> usize { 400 }

impl ViViTConfig {
    pub fn head_dim(&self) -> usize { self.hidden_size / self.num_attention_heads }
    pub fn num_patches(&self) -> usize {
        let t = self.tubelet_size[0];
        let h = self.tubelet_size[1];
        let w = self.tubelet_size[2];
        (self.num_frames / t) * (self.image_size / h) * (self.image_size / w)
    }
    pub fn seq_len(&self) -> usize { 1 + self.num_patches() } // CLS + patches
    pub fn tubelet_dim(&self) -> usize { 3 * self.tubelet_size[0] * self.tubelet_size[1] * self.tubelet_size[2] }
}

impl Default for ViViTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            image_size: 224,
            num_frames: 32,
            tubelet_size: vec![2, 16, 16],
            num_labels: 400,
            layer_norm_eps: 1e-6,
        }
    }
}

/// Load config from a JSON file.
pub fn load_config<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let config: T = serde_json::from_str(&text)?;
    Ok(config)
}

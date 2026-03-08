//! Binary save/load for QORA-Vision models.
//!
//! Format: magic "QVIS" + version(u32) + model_type(u8) + weights

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::gemv::{self, Weight};

const MAGIC: &[u8; 4] = b"QVIS";
const VERSION: u32 = 2;

// model_type: 0 = SigLIP vision, 1 = SigLIP text, 2 = ViViT
// (placeholder — can be expanded as needed)

fn write_header(w: &mut impl Write, format_id: u8) -> std::io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&[format_id])?;
    Ok(())
}

fn read_header(r: &mut impl Read) -> std::io::Result<u8> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Bad magic"));
    }
    let version = gemv::read_u32_io(r)?;
    if version != VERSION {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("Version mismatch: expected {VERSION}, got {version}")));
    }
    gemv::read_u8_io(r)
}

/// Save SigLIP vision weights to binary.
pub fn save_siglip_vision(
    vision: &crate::siglip::SigLIPVisionWeights,
    path: &Path,
) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = BufWriter::new(f);

    let format_id = match &vision.backbone.layers[0].q_proj {
        Weight::F16(_) => 0u8,
        Weight::Q4(_) => 1u8,
    };

    write_header(&mut w, format_id)?;

    // Patch embed
    gemv::write_f32_vec_io(&mut w, &vision.patch_embed_weight)?;
    gemv::write_f32_vec_io(&mut w, &vision.patch_embed_bias)?;
    gemv::write_f32_vec_io(&mut w, &vision.position_embedding)?;

    // Backbone layers
    w.write_all(&(vision.backbone.layers.len() as u32).to_le_bytes())?;
    for layer in &vision.backbone.layers {
        save_vit_layer(&mut w, layer)?;
    }

    // Post-LN
    gemv::write_f16_vec_io(&mut w, &vision.post_ln_gamma)?;
    gemv::write_f16_vec_io(&mut w, &vision.post_ln_beta)?;

    // MAP head
    gemv::write_f32_vec_io(&mut w, &vision.map_probe)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_q_weight)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_q_bias)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_k_weight)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_k_bias)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_v_weight)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_v_bias)?;
    gemv::write_weight_io(&mut w, &vision.map_out_weight)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_out_bias)?;
    gemv::write_f16_vec_io(&mut w, &vision.map_ln_gamma)?;
    gemv::write_f16_vec_io(&mut w, &vision.map_ln_beta)?;
    gemv::write_weight_io(&mut w, &vision.map_mlp_fc1)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_mlp_fc1_bias)?;
    gemv::write_weight_io(&mut w, &vision.map_mlp_fc2)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_mlp_fc2_bias)?;

    Ok(())
}

/// Save complete SigLIP model (vision + text + scale/bias) to binary.
pub fn save_siglip_full(
    vision: &crate::siglip::SigLIPVisionWeights,
    text: &crate::siglip::SigLIPTextWeights,
    logit_scale: f32,
    logit_bias: f32,
    path: &Path,
) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = BufWriter::new(f);

    let format_id = match &vision.backbone.layers[0].q_proj {
        Weight::F16(_) => 0u8,
        Weight::Q4(_) => 1u8,
    };

    write_header(&mut w, format_id)?;

    // Section marker: 0xFF = full model (vision + text)
    w.write_all(&[0xFFu8])?;

    // Logit scale and bias
    w.write_all(&logit_scale.to_le_bytes())?;
    w.write_all(&logit_bias.to_le_bytes())?;

    // === Vision encoder ===
    gemv::write_f32_vec_io(&mut w, &vision.patch_embed_weight)?;
    gemv::write_f32_vec_io(&mut w, &vision.patch_embed_bias)?;
    gemv::write_f32_vec_io(&mut w, &vision.position_embedding)?;

    w.write_all(&(vision.backbone.layers.len() as u32).to_le_bytes())?;
    for layer in &vision.backbone.layers {
        save_vit_layer(&mut w, layer)?;
    }

    gemv::write_f16_vec_io(&mut w, &vision.post_ln_gamma)?;
    gemv::write_f16_vec_io(&mut w, &vision.post_ln_beta)?;

    gemv::write_f32_vec_io(&mut w, &vision.map_probe)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_q_weight)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_q_bias)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_k_weight)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_k_bias)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_v_weight)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_v_bias)?;
    gemv::write_weight_io(&mut w, &vision.map_out_weight)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_out_bias)?;
    gemv::write_f16_vec_io(&mut w, &vision.map_ln_gamma)?;
    gemv::write_f16_vec_io(&mut w, &vision.map_ln_beta)?;
    gemv::write_weight_io(&mut w, &vision.map_mlp_fc1)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_mlp_fc1_bias)?;
    gemv::write_weight_io(&mut w, &vision.map_mlp_fc2)?;
    gemv::write_f32_vec_io(&mut w, &vision.map_mlp_fc2_bias)?;

    // === Text encoder ===
    gemv::write_weight_io(&mut w, &text.token_embedding)?;
    gemv::write_f32_vec_io(&mut w, &text.position_embedding)?;

    w.write_all(&(text.backbone.layers.len() as u32).to_le_bytes())?;
    for layer in &text.backbone.layers {
        save_vit_layer(&mut w, layer)?;
    }

    gemv::write_f16_vec_io(&mut w, &text.final_ln_gamma)?;
    gemv::write_f16_vec_io(&mut w, &text.final_ln_beta)?;
    gemv::write_weight_io(&mut w, &text.head_weight)?;
    gemv::write_f32_vec_io(&mut w, &text.head_bias)?;

    Ok(())
}

/// Load complete SigLIP model from binary.
pub fn load_siglip_full(path: &Path) -> std::io::Result<(
    crate::siglip::SigLIPVisionWeights,
    crate::siglip::SigLIPTextWeights,
    f32, // logit_scale
    f32, // logit_bias
)> {
    let f = std::fs::File::open(path)?;
    let mut r = BufReader::new(f);
    let format_id = read_header(&mut r)?;

    // Section marker
    let marker = gemv::read_u8_io(&mut r)?;
    if marker != 0xFF {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            "Not a full SigLIP binary (missing text encoder). Re-save with --save."));
    }

    // Logit scale and bias
    let logit_scale = {
        let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; f32::from_le_bytes(buf)
    };
    let logit_bias = {
        let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; f32::from_le_bytes(buf)
    };

    // === Vision encoder ===
    let patch_embed_weight = gemv::read_f32_vec_io(&mut r)?;
    let patch_embed_bias = gemv::read_f32_vec_io(&mut r)?;
    let position_embedding = gemv::read_f32_vec_io(&mut r)?;

    let num_layers = gemv::read_u32_io(&mut r)? as usize;
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers.push(load_vit_layer(&mut r, format_id)?);
    }

    let post_ln_gamma = gemv::read_f16_vec_io(&mut r)?;
    let post_ln_beta = gemv::read_f16_vec_io(&mut r)?;

    let map_probe = gemv::read_f32_vec_io(&mut r)?;
    let map_q_weight = gemv::read_f32_vec_io(&mut r)?;
    let map_q_bias = gemv::read_f32_vec_io(&mut r)?;
    let map_k_weight = gemv::read_f32_vec_io(&mut r)?;
    let map_k_bias = gemv::read_f32_vec_io(&mut r)?;
    let map_v_weight = gemv::read_f32_vec_io(&mut r)?;
    let map_v_bias = gemv::read_f32_vec_io(&mut r)?;
    let map_out_weight = gemv::read_weight_io(&mut r, format_id)?;
    let map_out_bias = gemv::read_f32_vec_io(&mut r)?;
    let map_ln_gamma = gemv::read_f16_vec_io(&mut r)?;
    let map_ln_beta = gemv::read_f16_vec_io(&mut r)?;
    let map_mlp_fc1 = gemv::read_weight_io(&mut r, format_id)?;
    let map_mlp_fc1_bias = gemv::read_f32_vec_io(&mut r)?;
    let map_mlp_fc2 = gemv::read_weight_io(&mut r, format_id)?;
    let map_mlp_fc2_bias = gemv::read_f32_vec_io(&mut r)?;

    let vision = crate::siglip::SigLIPVisionWeights {
        patch_embed_weight,
        patch_embed_bias,
        position_embedding,
        backbone: crate::vit::ViTBackbone {
            layers,
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            intermediate_size: 3072,
            ln_eps: 1e-6,
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
    let token_embedding = gemv::read_weight_io(&mut r, format_id)?;
    let text_position_embedding = gemv::read_f32_vec_io(&mut r)?;

    let text_num_layers = gemv::read_u32_io(&mut r)? as usize;
    let mut text_layers = Vec::with_capacity(text_num_layers);
    for _ in 0..text_num_layers {
        text_layers.push(load_vit_layer(&mut r, format_id)?);
    }

    let text_final_ln_gamma = gemv::read_f16_vec_io(&mut r)?;
    let text_final_ln_beta = gemv::read_f16_vec_io(&mut r)?;
    let head_weight = gemv::read_weight_io(&mut r, format_id)?;
    let head_bias = gemv::read_f32_vec_io(&mut r)?;

    let text = crate::siglip::SigLIPTextWeights {
        token_embedding,
        position_embedding: text_position_embedding,
        backbone: crate::vit::ViTBackbone {
            layers: text_layers,
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            intermediate_size: 3072,
            ln_eps: 1e-6,
        },
        final_ln_gamma: text_final_ln_gamma,
        final_ln_beta: text_final_ln_beta,
        head_weight,
        head_bias,
        vocab_size: 256000,
        max_position: 64,
    };

    Ok((vision, text, logit_scale, logit_bias))
}

/// Save ViViT weights to binary.
pub fn save_vivit(
    vivit: &crate::vivit::ViViTWeights,
    path: &Path,
) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = BufWriter::new(f);

    let format_id = match &vivit.backbone.layers[0].q_proj {
        Weight::F16(_) => 0u8,
        Weight::Q4(_) => 1u8,
    };

    write_header(&mut w, format_id)?;

    // Tubelet embed
    gemv::write_f32_vec_io(&mut w, &vivit.tubelet_weight)?;
    gemv::write_f32_vec_io(&mut w, &vivit.tubelet_bias)?;
    gemv::write_f32_vec_io(&mut w, &vivit.cls_token)?;
    gemv::write_f32_vec_io(&mut w, &vivit.position_embedding)?;

    // Backbone layers
    w.write_all(&(vivit.backbone.layers.len() as u32).to_le_bytes())?;
    for layer in &vivit.backbone.layers {
        save_vit_layer(&mut w, layer)?;
    }

    // Final LN + classifier
    gemv::write_f16_vec_io(&mut w, &vivit.final_ln_gamma)?;
    gemv::write_f16_vec_io(&mut w, &vivit.final_ln_beta)?;
    gemv::write_weight_io(&mut w, &vivit.classifier_weight)?;
    gemv::write_f32_vec_io(&mut w, &vivit.classifier_bias)?;

    Ok(())
}

fn save_vit_layer(w: &mut impl Write, layer: &crate::vit::ViTLayerWeights) -> std::io::Result<()> {
    gemv::write_weight_io(w, &layer.q_proj)?;
    gemv::write_weight_io(w, &layer.k_proj)?;
    gemv::write_weight_io(w, &layer.v_proj)?;
    gemv::write_weight_io(w, &layer.o_proj)?;
    gemv::write_f32_vec_io(w, &layer.q_bias)?;
    gemv::write_f32_vec_io(w, &layer.k_bias)?;
    gemv::write_f32_vec_io(w, &layer.v_bias)?;
    gemv::write_f32_vec_io(w, &layer.o_bias)?;
    gemv::write_f16_vec_io(w, &layer.ln1_gamma)?;
    gemv::write_f16_vec_io(w, &layer.ln1_beta)?;
    gemv::write_f16_vec_io(w, &layer.ln2_gamma)?;
    gemv::write_f16_vec_io(w, &layer.ln2_beta)?;
    gemv::write_weight_io(w, &layer.mlp_fc1)?;
    gemv::write_f32_vec_io(w, &layer.mlp_fc1_bias)?;
    gemv::write_weight_io(w, &layer.mlp_fc2)?;
    gemv::write_f32_vec_io(w, &layer.mlp_fc2_bias)?;
    Ok(())
}

fn load_vit_layer(r: &mut impl Read, format_id: u8) -> std::io::Result<crate::vit::ViTLayerWeights> {
    Ok(crate::vit::ViTLayerWeights {
        q_proj: gemv::read_weight_io(r, format_id)?,
        k_proj: gemv::read_weight_io(r, format_id)?,
        v_proj: gemv::read_weight_io(r, format_id)?,
        o_proj: gemv::read_weight_io(r, format_id)?,
        q_bias: gemv::read_f32_vec_io(r)?,
        k_bias: gemv::read_f32_vec_io(r)?,
        v_bias: gemv::read_f32_vec_io(r)?,
        o_bias: gemv::read_f32_vec_io(r)?,
        ln1_gamma: gemv::read_f16_vec_io(r)?,
        ln1_beta: gemv::read_f16_vec_io(r)?,
        ln2_gamma: gemv::read_f16_vec_io(r)?,
        ln2_beta: gemv::read_f16_vec_io(r)?,
        mlp_fc1: gemv::read_weight_io(r, format_id)?,
        mlp_fc1_bias: gemv::read_f32_vec_io(r)?,
        mlp_fc2: gemv::read_weight_io(r, format_id)?,
        mlp_fc2_bias: gemv::read_f32_vec_io(r)?,
    })
}

/// Load SigLIP vision from binary.
pub fn load_siglip_vision(path: &Path) -> std::io::Result<crate::siglip::SigLIPVisionWeights> {
    let f = std::fs::File::open(path)?;
    let mut r = BufReader::new(f);
    let format_id = read_header(&mut r)?;

    let patch_embed_weight = gemv::read_f32_vec_io(&mut r)?;
    let patch_embed_bias = gemv::read_f32_vec_io(&mut r)?;
    let position_embedding = gemv::read_f32_vec_io(&mut r)?;

    let num_layers = gemv::read_u32_io(&mut r)? as usize;
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers.push(load_vit_layer(&mut r, format_id)?);
    }

    let post_ln_gamma = gemv::read_f16_vec_io(&mut r)?;
    let post_ln_beta = gemv::read_f16_vec_io(&mut r)?;

    let map_probe = gemv::read_f32_vec_io(&mut r)?;
    let map_q_weight = gemv::read_f32_vec_io(&mut r)?;
    let map_q_bias = gemv::read_f32_vec_io(&mut r)?;
    let map_k_weight = gemv::read_f32_vec_io(&mut r)?;
    let map_k_bias = gemv::read_f32_vec_io(&mut r)?;
    let map_v_weight = gemv::read_f32_vec_io(&mut r)?;
    let map_v_bias = gemv::read_f32_vec_io(&mut r)?;
    let map_out_weight = gemv::read_weight_io(&mut r, format_id)?;
    let map_out_bias = gemv::read_f32_vec_io(&mut r)?;
    let map_ln_gamma = gemv::read_f16_vec_io(&mut r)?;
    let map_ln_beta = gemv::read_f16_vec_io(&mut r)?;
    let map_mlp_fc1 = gemv::read_weight_io(&mut r, format_id)?;
    let map_mlp_fc1_bias = gemv::read_f32_vec_io(&mut r)?;
    let map_mlp_fc2 = gemv::read_weight_io(&mut r, format_id)?;
    let map_mlp_fc2_bias = gemv::read_f32_vec_io(&mut r)?;

    Ok(crate::siglip::SigLIPVisionWeights {
        patch_embed_weight,
        patch_embed_bias,
        position_embedding,
        backbone: crate::vit::ViTBackbone {
            layers,
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            intermediate_size: 3072,
            ln_eps: 1e-6,
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
    })
}

/// Load ViViT from binary.
pub fn load_vivit(path: &Path) -> std::io::Result<crate::vivit::ViViTWeights> {
    let f = std::fs::File::open(path)?;
    let mut r = BufReader::new(f);
    let format_id = read_header(&mut r)?;

    let tubelet_weight = gemv::read_f32_vec_io(&mut r)?;
    let tubelet_bias = gemv::read_f32_vec_io(&mut r)?;
    let cls_token = gemv::read_f32_vec_io(&mut r)?;
    let position_embedding = gemv::read_f32_vec_io(&mut r)?;

    let num_layers = gemv::read_u32_io(&mut r)? as usize;
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers.push(load_vit_layer(&mut r, format_id)?);
    }

    let final_ln_gamma = gemv::read_f16_vec_io(&mut r)?;
    let final_ln_beta = gemv::read_f16_vec_io(&mut r)?;
    let classifier_weight = gemv::read_weight_io(&mut r, format_id)?;
    let classifier_bias = gemv::read_f32_vec_io(&mut r)?;

    Ok(crate::vivit::ViViTWeights {
        tubelet_weight,
        tubelet_bias,
        cls_token,
        position_embedding,
        backbone: crate::vit::ViTBackbone {
            layers,
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            intermediate_size: 3072,
            ln_eps: 1e-6,
        },
        final_ln_gamma,
        final_ln_beta,
        classifier_weight,
        classifier_bias,
        num_patches: 3136,
        num_frames: 32,
        tubelet_size: [2, 16, 16],
    })
}

//! Image I/O: load, resize, normalize for SigLIP and ViViT.

use std::path::Path;

/// Load an image, resize to 224x224, normalize to [-1, 1].
/// Returns [3, height, width] in CHW float32 layout.
pub fn load_image_siglip(path: &Path, target_size: u32) -> Vec<f32> {
    let img = image::open(path).expect("Failed to open image");
    let img = img.resize_exact(target_size, target_size, image::imageops::FilterType::Triangle);
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    let mut output = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                // SigLIP: pixel / 127.5 - 1.0  maps [0,255] to [-1,1]
                output[c * h * w + y * w + x] = pixel[c] as f32 / 127.5 - 1.0;
            }
        }
    }
    output
}

/// Load an image for ViViT: resize shortest edge to target, center crop.
/// Returns [3, height, width] in CHW float32 layout, normalized.
pub fn load_image_vivit(path: &Path, target_size: u32) -> Vec<f32> {
    let img = image::open(path).expect("Failed to open image");
    let (w, h) = (img.width(), img.height());

    // Resize shortest edge to target_size
    let scale = target_size as f32 / w.min(h) as f32;
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;
    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);

    // Center crop
    let x_off = (new_w - target_size) / 2;
    let y_off = (new_h - target_size) / 2;
    let cropped = resized.crop_imm(x_off, y_off, target_size, target_size);
    let rgb = cropped.to_rgb8();
    let (cw, ch) = (rgb.width() as usize, rgb.height() as usize);

    let mut output = vec![0.0f32; 3 * ch * cw];
    for y in 0..ch {
        for x in 0..cw {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                // ViViT: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1.0  (same range [-1,1])
                output[c * ch * cw + y * cw + x] = pixel[c] as f32 / 127.5 - 1.0;
            }
        }
    }
    output
}

//! Video frame loading for ViViT.
//!
//! Supports two modes:
//! 1. Directory of pre-extracted frames (JPEG/PNG)
//! 2. Video file (requires ffmpeg in PATH)

use std::path::Path;

/// Load video frames from a directory of images.
/// Frames should be named in sortable order (e.g., frame_0000.jpg).
/// Returns [3, num_frames, 224, 224] in CTHW format, normalized.
pub fn load_frames_from_directory(dir: &Path, num_frames: usize, target_size: u32) -> Vec<f32> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .expect("Failed to read frame directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_lowercase();
            name.ends_with(".jpg") || name.ends_with(".jpeg") || name.ends_with(".png")
        })
        .map(|e| e.path())
        .collect();
    paths.sort();

    if paths.is_empty() {
        panic!("No image frames found in {}", dir.display());
    }

    // Sample uniformly if we have more frames than needed
    let selected: Vec<_> = if paths.len() >= num_frames {
        let step = paths.len() as f64 / num_frames as f64;
        (0..num_frames)
            .map(|i| paths[(i as f64 * step) as usize].clone())
            .collect()
    } else {
        // Repeat last frame to fill
        let mut selected = paths.clone();
        while selected.len() < num_frames {
            selected.push(paths.last().unwrap().clone());
        }
        selected
    };

    eprintln!("  Loading {} frames from {}...", num_frames, dir.display());

    // Load each frame as [3, H, W] and stack to [3, T, H, W]
    let ts = target_size as usize;
    let mut video = vec![0.0f32; 3 * num_frames * ts * ts];

    for (t, path) in selected.iter().enumerate() {
        let frame = crate::image_io::load_image_vivit(path, target_size);
        // frame is [3, ts, ts], copy into video[c, t, y, x]
        for c in 0..3 {
            for y in 0..ts {
                for x in 0..ts {
                    let src = c * ts * ts + y * ts + x;
                    let dst = c * num_frames * ts * ts + t * ts * ts + y * ts + x;
                    video[dst] = frame[src];
                }
            }
        }
    }

    video
}

/// Extract frames from a video file using ffmpeg, then load.
/// Requires ffmpeg in PATH.
pub fn load_frames_from_video(video_path: &Path, num_frames: usize, target_size: u32) -> Vec<f32> {
    let tmp_dir = std::env::temp_dir().join("qora_vision_frames");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).expect("Failed to create temp dir");

    eprintln!("  Extracting {num_frames} frames from {}...", video_path.display());

    // Extract frames with ffmpeg
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-i", &video_path.to_string_lossy(),
            "-vf", &format!("fps=8,scale={}:{}", target_size, target_size),
            "-frames:v", &num_frames.to_string(),
            &tmp_dir.join("frame_%04d.png").to_string_lossy(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .expect("Failed to run ffmpeg. Make sure it's installed and in PATH.");

    if !status.success() {
        panic!("ffmpeg failed with status: {}", status);
    }

    let result = load_frames_from_directory(&tmp_dir, num_frames, target_size);

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);
    result
}

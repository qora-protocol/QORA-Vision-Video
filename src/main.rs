use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("QORA-Vision — Pure Rust Vision Models");
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  qora-vision siglip --image photo.jpg [--labels cat,dog]");
        eprintln!("  qora-vision vivit  --frames ./dir/");
        eprintln!("  qora-vision vivit  --video clip.mp4");
        eprintln!();
        eprintln!("Common flags: --load <path>, --cpu");
        std::process::exit(1);
    }

    let subcommand = args[1].as_str();

    // Parse common flags
    let mut load_path = PathBuf::from("model.qora-vision");
    #[cfg(any(feature = "gpu", feature = "gpu-metal"))]
    let force_cpu = args.iter().any(|a| a == "--cpu");

    // SigLIP-specific
    let mut image_path: Option<PathBuf> = None;
    let mut labels: Option<String> = None;
    let mut text: Option<String> = None;

    // ViViT-specific
    let mut frames_dir: Option<PathBuf> = None;
    let mut video_path: Option<PathBuf> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--load" => { if i + 1 < args.len() { load_path = PathBuf::from(&args[i + 1]); i += 1; } }
            "--image" => { if i + 1 < args.len() { image_path = Some(PathBuf::from(&args[i + 1])); i += 1; } }
            "--labels" => { if i + 1 < args.len() { labels = Some(args[i + 1].clone()); i += 1; } }
            "--text" => { if i + 1 < args.len() { text = Some(args[i + 1].clone()); i += 1; } }
            "--frames" => { if i + 1 < args.len() { frames_dir = Some(PathBuf::from(&args[i + 1])); i += 1; } }
            "--video" => { if i + 1 < args.len() { video_path = Some(PathBuf::from(&args[i + 1])); i += 1; } }
            "--cpu" => {} // handled above
            _ => {}
        }
        i += 1;
    }

    match subcommand {
        "siglip" => run_siglip(load_path, image_path, labels, text,
            #[cfg(any(feature = "gpu", feature = "gpu-metal"))] force_cpu),
        "vivit" => run_vivit(load_path, frames_dir, video_path,
            #[cfg(any(feature = "gpu", feature = "gpu-metal"))] force_cpu),
        _ => {
            eprintln!("Unknown subcommand: {subcommand}");
            eprintln!("Use 'siglip' or 'vivit'");
            std::process::exit(1);
        }
    }
}

fn run_siglip(
    load_path: PathBuf, image_path: Option<PathBuf>,
    labels: Option<String>, text: Option<String>,
    #[cfg(any(feature = "gpu", feature = "gpu-metal"))] force_cpu: bool,
) {
    eprintln!("QORA-Vision — Image Encoder (Base, 224px)");
    eprintln!();

    let t0 = Instant::now();

    // Load from .qora-vision binary
    eprintln!("Loading from {}...", load_path.display());
    let (vision, text_weights, logit_scale, logit_bias) =
        match qora_vision::save::load_siglip_full(&load_path) {
            Ok((v, t, ls, lb)) => {
                let vmb = v.memory_bytes() / (1024 * 1024);
                let tmb = t.memory_bytes() / (1024 * 1024);
                eprintln!("Loaded in {:.1?} (vision: {vmb} MB, text: {tmb} MB)", t0.elapsed());
                (v, Some(t), ls, lb)
            }
            Err(_) => {
                // Fall back to vision-only binary
                let vision = qora_vision::save::load_siglip_vision(&load_path)
                    .expect("Failed to load .qora-vision model");
                eprintln!("Vision loaded in {:.1?}", t0.elapsed());
                (vision, None, 0.0f32, 0.0f32)
            }
        };

    // Look for tokenizer next to the binary
    let base_path = load_path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();

    // Process image
    if let Some(ref img_path) = image_path {
        eprintln!("Processing image: {}", img_path.display());
        let t_img = Instant::now();
        let image = qora_vision::image_io::load_image_siglip(img_path, 224);
        eprintln!("Image loaded in {:.1?}", t_img.elapsed());

        // === Try GPU inference ===
        #[cfg(any(feature = "gpu", feature = "gpu-metal"))]
        if !force_cpu {
            eprintln!("Attempting GPU inference...");

            // Prepare labels and tokenizer for GPU
            let label_list: Vec<&str> = labels.as_ref()
                .map(|s| s.split(',').map(|l| l.trim()).collect())
                .unwrap_or_default();

            let tokenizer = if !label_list.is_empty() {
                let tok_path = base_path.join("tokenizer.json");
                Some(qora_vision::tokenizer::TextTokenizer::from_file(&tok_path)
                    .expect("Failed to load tokenizer"))
            } else {
                None
            };

            match qora_vision::gpu_inference::run_siglip_gpu(
                &image,
                &vision,
                text_weights.as_ref(),
                &label_list,
                tokenizer.as_ref(),
                logit_scale,
                logit_bias,
            ) {
                Ok((image_embed, scores)) => {
                    let norm: f32 = image_embed.iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!("Embedding: dim={}, L2 norm={:.4}", image_embed.len(), norm);
                    eprintln!("Top-5 dims: {:.4} {:.4} {:.4} {:.4} {:.4}",
                        image_embed[0], image_embed[1], image_embed[2], image_embed[3], image_embed[4]);

                    if !scores.is_empty() {
                        eprintln!("\nZero-shot classification ({} labels):", scores.len());
                        for (label, score) in &scores {
                            eprintln!("  {label}: {score:.4}");
                        }
                        let best = scores.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
                        eprintln!("Best: {} ({:.4})", best.0, best.1);
                    }

                    // Handle --text on GPU result
                    if let (Some(ref txt), Some(ref tw)) = (&text, &text_weights) {
                        let tok_path = base_path.join("tokenizer.json");
                        let tok = tokenizer.unwrap_or_else(|| {
                            qora_vision::tokenizer::TextTokenizer::from_file(&tok_path)
                                .expect("Failed to load tokenizer")
                        });
                        let token_ids = tok.encode(txt);
                        let mut text_embed = qora_vision::siglip::siglip_text_forward(&token_ids, tw);
                        qora_vision::siglip::l2_normalize(&mut text_embed);
                        let sim = qora_vision::siglip::cosine_similarity(&image_embed, &text_embed);
                        eprintln!("Text: \"{txt}\" → cosine similarity: {sim:.4}");
                    }

                    eprintln!("\nTotal: {:.1?}", t0.elapsed());
                    return;
                }
                Err(e) => eprintln!("GPU not available ({}), falling back to CPU", e),
            }
        }

        // === CPU inference ===
        let t_fwd = Instant::now();
        let mut image_embed = qora_vision::siglip::siglip_vision_forward(&image, &vision);
        eprintln!("Vision forward in {:.1?}", t_fwd.elapsed());

        // L2 normalize
        qora_vision::siglip::l2_normalize(&mut image_embed);

        // Print embedding stats
        let norm: f32 = image_embed.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("Embedding: dim={}, L2 norm={:.4}", image_embed.len(), norm);
        eprintln!("Top-5 dims: {:.4} {:.4} {:.4} {:.4} {:.4}",
            image_embed[0], image_embed[1], image_embed[2], image_embed[3], image_embed[4]);

        // Zero-shot classification if labels provided
        if let (Some(ref label_str), Some(ref tw)) = (&labels, &text_weights) {
            let label_list: Vec<&str> = label_str.split(',').map(|s| s.trim()).collect();
            eprintln!("\nZero-shot classification ({} labels):", label_list.len());

            let tok_path = base_path.join("tokenizer.json");
            let tokenizer = qora_vision::tokenizer::TextTokenizer::from_file(&tok_path)
                .expect("Failed to load tokenizer");

            let mut scores = Vec::new();
            for label in &label_list {
                let prompt = format!("This is a photo of {label}.");
                let token_ids = tokenizer.encode(&prompt);
                let mut text_embed = qora_vision::siglip::siglip_text_forward(&token_ids, tw);
                qora_vision::siglip::l2_normalize(&mut text_embed);
                let score = qora_vision::siglip::siglip_score(&image_embed, &text_embed, logit_scale, logit_bias);
                scores.push(score);
                eprintln!("  {label}: {score:.4}");
            }

            let best = scores.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
            eprintln!("Best: {} ({:.4})", label_list[best.0], best.1);
        }

        // Text embedding if --text provided
        if let (Some(ref txt), Some(ref tw)) = (&text, &text_weights) {
            let tok_path = base_path.join("tokenizer.json");
            let tokenizer = qora_vision::tokenizer::TextTokenizer::from_file(&tok_path)
                .expect("Failed to load tokenizer");
            let token_ids = tokenizer.encode(txt);
            let mut text_embed = qora_vision::siglip::siglip_text_forward(&token_ids, tw);
            qora_vision::siglip::l2_normalize(&mut text_embed);
            let sim = qora_vision::siglip::cosine_similarity(&image_embed, &text_embed);
            eprintln!("Text: \"{txt}\" → cosine similarity: {sim:.4}");
        }
    }

    eprintln!("\nTotal: {:.1?}", t0.elapsed());
}

fn run_vivit(
    load_path: PathBuf, frames_dir: Option<PathBuf>, video_path: Option<PathBuf>,
    #[cfg(any(feature = "gpu", feature = "gpu-metal"))] force_cpu: bool,
) {
    eprintln!("QORA-Vision — Video Encoder (Base, 16x2, Kinetics-400)");
    eprintln!();

    let t0 = Instant::now();

    // Load from .qora-vision binary
    eprintln!("Loading from {}...", load_path.display());
    let weights = qora_vision::save::load_vivit(&load_path)
        .expect("Failed to load .qora-vision model");
    eprintln!("Loaded in {:.1?}", t0.elapsed());

    // Load video
    let video = if let Some(ref dir) = frames_dir {
        qora_vision::video::load_frames_from_directory(dir, 32, 224)
    } else if let Some(ref vp) = video_path {
        qora_vision::video::load_frames_from_video(vp, 32, 224)
    } else {
        eprintln!("No video input specified. Use --frames <dir> or --video <file>");
        eprintln!("\nTotal: {:.1?}", t0.elapsed());
        return;
    };

    eprintln!("Video: {} values ({:.1} MB)", video.len(), video.len() as f32 * 4.0 / 1024.0 / 1024.0);

    // === Try GPU inference ===
    #[cfg(any(feature = "gpu", feature = "gpu-metal"))]
    if !force_cpu {
        eprintln!("Attempting GPU inference...");
        match qora_vision::gpu_inference::run_vivit_gpu(&video, &weights) {
            Ok((embedding, logits)) => {
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("Embedding: dim={}, L2 norm={:.4}", embedding.len(), norm);

                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

                eprintln!("\nTop-5 predictions:");
                for (rank, &(idx, score)) in indexed.iter().take(5).enumerate() {
                    eprintln!("  #{}: class {idx} (score: {score:.4})", rank + 1);
                }

                eprintln!("\nTotal: {:.1?}", t0.elapsed());
                return;
            }
            Err(e) => eprintln!("GPU not available ({}), falling back to CPU", e),
        }
    }

    // === CPU inference ===
    let t_fwd = Instant::now();
    let (embedding, logits) = qora_vision::vivit::vivit_forward(&video, &weights);
    eprintln!("Forward pass in {:.1?}", t_fwd.elapsed());

    // Print embedding stats
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("Embedding: dim={}, L2 norm={:.4}", embedding.len(), norm);

    // Top-5 predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    eprintln!("\nTop-5 predictions:");
    for (rank, &(idx, score)) in indexed.iter().take(5).enumerate() {
        eprintln!("  #{}: class {idx} (score: {score:.4})", rank + 1);
    }

    eprintln!("\nTotal: {:.1?}", t0.elapsed());
}

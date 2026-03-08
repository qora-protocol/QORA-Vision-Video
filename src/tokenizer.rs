//! SigLIP text tokenizer.
//!
//! Uses the `tokenizers` crate to load tokenizer.json from the model directory.

use std::path::Path;

pub struct TextTokenizer {
    inner: tokenizers::Tokenizer,
}

impl TextTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;
        Ok(Self { inner: tokenizer })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self.inner.encode(text, true)
            .expect("Failed to encode text");
        encoding.get_ids().to_vec()
    }
}

use anyhow::Result;
use std::sync::Arc;
use tiktoken_rs::{CoreBPE, cl100k_base, o200k_base};

pub struct Tokenizer {
    encoder: Arc<CoreBPE>,
    model_type: ModelType,
}

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    // GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    Cl100k,
    // GPT-4o models
    O200k,
}

impl Tokenizer {
    pub fn new(model: &str) -> Result<Self> {
        let (encoder, model_type) = if model.contains("gpt-4o") {
            (Arc::new(o200k_base()?), ModelType::O200k)
        } else {
            // Default to cl100k for most models (GPT-3.5, GPT-4, etc.)
            (Arc::new(cl100k_base()?), ModelType::Cl100k)
        };

        Ok(Self {
            encoder,
            model_type,
        })
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        // Note: This counts tokens in the raw text content only.
        // When using chat APIs, additional tokens are added for:
        // - Chat format markers (e.g., <|im_start|>, <|im_end|>)
        // - Role indicators (e.g., "user", "assistant")
        // - Other protocol overhead
        // So the actual tokens sent to the API will be higher than this count.
        self.encoder.encode_with_special_tokens(text).len()
    }

    pub fn model_type(&self) -> ModelType {
        self.model_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_counting() {
        let tokenizer = Tokenizer::new("gpt-3.5-turbo").unwrap();

        // Test basic text
        let count = tokenizer.count_tokens("Hello, world!");
        assert!(count > 0);

        // Test that whitespace counting differs from token counting
        let text = "This is a test";
        let word_count = text.split_whitespace().count();
        let token_count = tokenizer.count_tokens(text);
        // Tokens and words are usually different
        println!("Words: {}, Tokens: {}", word_count, token_count);
    }
}

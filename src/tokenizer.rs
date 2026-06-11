use anyhow::Result;
use std::sync::Arc;
use tiktoken_rs::{CoreBPE, cl100k_base, o200k_base};

pub struct Tokenizer {
    encoder: Arc<CoreBPE>,
    model_type: ModelType,
    is_estimate: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    // GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    Cl100k,
    // GPT-4o and o-series models
    O200k,
}

impl ModelType {
    /// The tiktoken vocabulary name, for display.
    pub fn encoder_name(&self) -> &'static str {
        match self {
            ModelType::Cl100k => "cl100k_base",
            ModelType::O200k => "o200k_base",
        }
    }
}

/// Pick the tiktoken vocabulary for a model name and report whether the result is
/// only an *estimate*. tiktoken exactly matches OpenAI models; for anything else
/// (Llama, Qwen, Mistral, …) the real BPE differs, so counts are approximate.
pub fn select_model_type(model: &str) -> (ModelType, bool) {
    let m = model.to_lowercase();
    // Anchor o-series detection to the leaf name so local models that merely
    // start with "o1"/"o3" (e.g. "o3de-...") aren't mistaken for OpenAI models.
    let leaf = m.rsplit('/').next().unwrap_or(m.as_str());
    let is_o_series = ["o1", "o3", "o4"]
        .iter()
        .any(|p| leaf == *p || leaf.starts_with(&format!("{p}-")));
    if m.contains("gpt-4o") || is_o_series || m.contains("o200k") {
        (ModelType::O200k, false)
    } else if m.contains("gpt-4")
        || m.contains("gpt-3.5")
        || m.contains("gpt-35")
        || m.contains("cl100k")
        || m.contains("text-embedding")
        || m.contains("davinci")
    {
        (ModelType::Cl100k, false)
    } else {
        // Fall back to cl100k, but the count is only an estimate for this model.
        (ModelType::Cl100k, true)
    }
}

impl Tokenizer {
    pub fn new(model: &str) -> Result<Self> {
        let (model_type, is_estimate) = select_model_type(model);
        let encoder = match model_type {
            ModelType::O200k => Arc::new(o200k_base()?),
            ModelType::Cl100k => Arc::new(cl100k_base()?),
        };

        if is_estimate {
            log::warn!(
                "Tokenizer for model '{model}' falls back to tiktoken {} — the real \
                 tokenizer differs, so locally counted token counts (synthetic prompt \
                 lengths, and any per-token metric without server usage) are estimates. \
                 Server-reported token counts are used where available.",
                model_type.encoder_name()
            );
        }

        Ok(Self {
            encoder,
            model_type,
            is_estimate,
        })
    }

    /// Whether local token counts are estimates (model isn't an OpenAI tiktoken model).
    pub fn is_estimate(&self) -> bool {
        self.is_estimate
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

    /// Truncate text to at most `max_tokens` tokens. Encodes once and decodes the prefix,
    /// avoiding the O(log N) repeated tokenization of a binary search.
    pub fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> String {
        let tokens = self.encoder.encode_with_special_tokens(text);
        if tokens.len() <= max_tokens {
            return text.to_string();
        }
        self.encoder
            .decode(tokens[..max_tokens].to_vec())
            .unwrap_or_else(|_| text.to_string())
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

    #[test]
    fn openai_models_are_exact_not_estimates() {
        // Known OpenAI families map to their real tiktoken vocab.
        assert!(!Tokenizer::new("gpt-4o").unwrap().is_estimate());
        assert!(!Tokenizer::new("gpt-4o-mini").unwrap().is_estimate());
        assert!(!Tokenizer::new("gpt-4").unwrap().is_estimate());
        assert!(!Tokenizer::new("gpt-3.5-turbo").unwrap().is_estimate());
    }

    #[test]
    fn non_openai_models_are_flagged_as_estimates() {
        // Local servers serve Llama/Qwen/Mistral etc., whose real BPE differs
        // from tiktoken — local counts are only estimates.
        assert!(
            Tokenizer::new("meta-llama/Llama-3.1-8B-Instruct")
                .unwrap()
                .is_estimate()
        );
        assert!(Tokenizer::new("Qwen/Qwen2.5-7B").unwrap().is_estimate());
        assert!(
            Tokenizer::new("mistralai/Mistral-7B-v0.3")
                .unwrap()
                .is_estimate()
        );
    }

    #[test]
    fn o_series_detection_is_anchored_not_a_loose_prefix() {
        // Real OpenAI o-series → exact.
        assert!(!select_model_type("o1").1);
        assert!(!select_model_type("o1-mini").1);
        assert!(!select_model_type("openai/o3-mini").1);
        // Local models that merely start with "o1"/"o3" must NOT be treated as
        // exact OpenAI models.
        assert!(select_model_type("o3de-game-llm").1);
        assert!(select_model_type("o1de-7b").1);
    }
}

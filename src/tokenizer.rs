use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use tiktoken_rs::{CoreBPE, cl100k_base, o200k_base};

/// Counts tokens for prompt sizing/truncation. Backed by, in priority order: the
/// model's real `tokenizer.json` (exact), a tiktoken count calibrated against the
/// server's `/tokenize` (confident estimate), or raw tiktoken (rough estimate).
pub struct Tokenizer {
    backend: Backend,
}

enum Backend {
    /// tiktoken vocabulary; `estimate` is true for non-OpenAI models.
    Tiktoken {
        encoder: Arc<CoreBPE>,
        model_type: ModelType,
        estimate: bool,
    },
    /// tiktoken count scaled by a ratio (real/tiktoken) calibrated against the
    /// server's `/tokenize` endpoint — a confident estimate without per-prompt calls.
    Calibrated { encoder: Arc<CoreBPE>, ratio: f64 },
    /// The model's real tokenizer loaded from `tokenizer.json` (exact).
    Exact {
        inner: Arc<tokenizers::Tokenizer>,
        name: String,
    },
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

/// tiktoken vocabulary for a model name (used by the tiktoken/calibrated backends).
fn tiktoken_for(model: &str) -> Result<(Arc<CoreBPE>, ModelType, bool)> {
    let (model_type, estimate) = select_model_type(model);
    let encoder = match model_type {
        ModelType::O200k => Arc::new(o200k_base()?),
        ModelType::Cl100k => Arc::new(cl100k_base()?),
    };
    Ok((encoder, model_type, estimate))
}

/// Compute the calibration ratio (real tokens / tiktoken tokens) from paired
/// sample counts. Returns 1.0 if there's no usable tiktoken total.
pub fn calibration_ratio(real_total: usize, tiktoken_total: usize) -> f64 {
    if tiktoken_total == 0 {
        1.0
    } else {
        real_total as f64 / tiktoken_total as f64
    }
}

impl Tokenizer {
    /// tiktoken backend, chosen by model name. Exact for OpenAI models, an estimate
    /// (with a warning) otherwise.
    pub fn new(model: &str) -> Result<Self> {
        let (encoder, model_type, estimate) = tiktoken_for(model)?;
        if estimate {
            log::warn!(
                "Tokenizer for model '{model}' falls back to tiktoken {} — the real \
                 tokenizer differs, so locally counted token counts (synthetic prompt \
                 lengths, and any per-token metric without server usage) are estimates. \
                 Configure `endpoint.tokenizer` (a tokenizer.json path or HF id) for exact \
                 counts; server-reported counts are used where available.",
                model_type.encoder_name()
            );
        }
        Ok(Self {
            backend: Backend::Tiktoken {
                encoder,
                model_type,
                estimate,
            },
        })
    }

    /// Exact tokenizer loaded from a local `tokenizer.json` file.
    pub fn from_tokenizer_json(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            anyhow::anyhow!("failed to load tokenizer.json from {}: {e}", path.display())
        })?;
        // Smoke-test that it can actually encode, so a broken tokenizer fails here
        // (and the caller falls back) rather than silently returning 0 tokens per
        // prompt and mis-sizing the whole run.
        inner.encode("tokenizer warmup", false).map_err(|e| {
            anyhow::anyhow!("tokenizer from {} failed to encode: {e}", path.display())
        })?;
        Ok(Self {
            backend: Backend::Exact {
                inner: Arc::new(inner),
                name: path.display().to_string(),
            },
        })
    }

    /// Calibrated estimate: tiktoken counts scaled by `ratio`. `base_model` picks
    /// the tiktoken vocabulary that the ratio was measured against.
    pub fn calibrated(base_model: &str, ratio: f64) -> Result<Self> {
        let (encoder, _, _) = tiktoken_for(base_model)?;
        Ok(Self {
            backend: Backend::Calibrated { encoder, ratio },
        })
    }

    /// Whether token counts are estimates (true for tiktoken-on-non-OpenAI and
    /// calibrated; false for an exact `tokenizer.json`).
    pub fn is_estimate(&self) -> bool {
        match &self.backend {
            Backend::Tiktoken { estimate, .. } => *estimate,
            Backend::Calibrated { .. } => true,
            Backend::Exact { .. } => false,
        }
    }

    /// Count tokens in raw text content (chat-template markers/role tokens are
    /// added by the server on top of this).
    pub fn count_tokens(&self, text: &str) -> usize {
        match &self.backend {
            Backend::Tiktoken { encoder, .. } => encoder.encode_with_special_tokens(text).len(),
            Backend::Calibrated { encoder, ratio } => {
                let raw = encoder.encode_with_special_tokens(text).len();
                (raw as f64 * ratio).round() as usize
            }
            Backend::Exact { inner, .. } => inner
                .encode(text, false)
                .map(|e| e.len())
                .unwrap_or_else(|_| 0),
        }
    }

    /// Truncate text to at most `max_tokens` tokens (in this tokenizer's units).
    pub fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> String {
        match &self.backend {
            Backend::Exact { inner, .. } => {
                let enc = match inner.encode(text, false) {
                    Ok(e) => e,
                    Err(_) => return text.to_string(),
                };
                let ids = enc.get_ids();
                if ids.len() <= max_tokens {
                    return text.to_string();
                }
                inner
                    .decode(&ids[..max_tokens], true)
                    .unwrap_or_else(|_| text.to_string())
            }
            Backend::Calibrated { encoder, ratio } => {
                // Cut in tiktoken units so count_tokens(result) ≈ max_tokens.
                let target = if *ratio > 0.0 {
                    (max_tokens as f64 / ratio).round() as usize
                } else {
                    max_tokens
                };
                truncate_tiktoken(encoder, text, target)
            }
            Backend::Tiktoken { encoder, .. } => truncate_tiktoken(encoder, text, max_tokens),
        }
    }

    /// Human-readable description of the active tokenizer, for the report.
    pub fn source_label(&self) -> String {
        match &self.backend {
            Backend::Tiktoken {
                model_type,
                estimate,
                ..
            } => {
                if *estimate {
                    format!("tiktoken {} (estimate)", model_type.encoder_name())
                } else {
                    format!("tiktoken {}", model_type.encoder_name())
                }
            }
            Backend::Calibrated { ratio, .. } => {
                format!("calibrated estimate (server /tokenize, ratio {ratio:.3})")
            }
            Backend::Exact { name, .. } => format!("exact ({name})"),
        }
    }
}

/// Truncate by tiktoken: encode once, decode the first `max_tokens` ids.
fn truncate_tiktoken(encoder: &CoreBPE, text: &str, max_tokens: usize) -> String {
    let tokens = encoder.encode_with_special_tokens(text);
    if tokens.len() <= max_tokens {
        return text.to_string();
    }
    encoder
        .decode(tokens[..max_tokens].to_vec())
        .unwrap_or_else(|_| text.to_string())
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
    fn calibration_ratio_is_real_over_tiktoken() {
        assert!((calibration_ratio(132, 100) - 1.32).abs() < 1e-9);
        // Guards: no tiktoken tokens → neutral ratio.
        assert_eq!(calibration_ratio(50, 0), 1.0);
        assert_eq!(calibration_ratio(0, 0), 1.0);
    }

    #[test]
    fn calibrated_scales_tiktoken_count() {
        let text = "The quick brown fox jumps over the lazy dog, repeatedly and verbosely.";
        let raw = Tokenizer::new("gpt-3.5-turbo").unwrap().count_tokens(text);
        let calibrated = Tokenizer::calibrated("gpt-3.5-turbo", 1.5).unwrap();
        assert_eq!(
            calibrated.count_tokens(text),
            (raw as f64 * 1.5).round() as usize
        );
        // Calibrated is a (confident) estimate, and labels itself as such.
        assert!(calibrated.is_estimate());
        assert!(calibrated.source_label().contains("calibrated"));
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

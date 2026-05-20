//! Synthetic data generation for benchmarking LLM servers.
//!
//! This module provides functionality to generate synthetic prompts with exact token counts,
//! similar to guidellm's approach. It uses the fake-rs library for random text generation
//! and tiktoken-rs for tokenization.

use std::sync::Arc;

use anyhow::Result;
use fake::Fake;
use fake::faker::lorem::en::*;
use log::warn;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::benchmark::{Conversation, Prompt, Workload};
use crate::config::SyntheticConfig;
use crate::tokenizer::Tokenizer;

/// Samples token counts from a Gaussian distribution with hard min/max bounds.
pub struct TokenDistribution {
    average: usize,
    stdev: Option<usize>,
    min: usize,
    max: usize,
    rng: StdRng,
}

impl TokenDistribution {
    /// Create a new token distribution.
    ///
    /// # Arguments
    /// * `average` - Average token count
    /// * `stdev` - Optional standard deviation for Gaussian sampling
    /// * `min` - Optional minimum (defaults to max(0, average - 5*stdev))
    /// * `max` - Optional maximum (defaults to average + 5*stdev)
    /// * `seed` - Random seed for reproducibility
    pub fn new(
        average: usize,
        stdev: Option<usize>,
        min: Option<usize>,
        max: Option<usize>,
        seed: u64,
    ) -> Self {
        let calc_min = min.unwrap_or_else(|| {
            if let Some(s) = stdev {
                average.saturating_sub(5 * s)
            } else {
                average
            }
        });

        let calc_max = max.unwrap_or_else(|| {
            if let Some(s) = stdev {
                average + 5 * s
            } else {
                average
            }
        });

        Self {
            average,
            stdev,
            min: calc_min,
            max: calc_max,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample a token count from the distribution.
    pub fn sample(&mut self) -> usize {
        if self.min == self.max {
            return self.min;
        }

        match self.stdev {
            None => {
                // Uniform distribution between min and max
                self.rng.gen_range(self.min..=self.max)
            }
            Some(0) => {
                // No variance
                self.average
            }
            Some(stdev) => {
                // Gaussian distribution with clamping
                let normal = Normal::new(self.average as f64, stdev as f64)
                    .expect("Failed to create normal distribution");
                let sample = normal.sample(&mut self.rng).round() as usize;
                sample.clamp(self.min, self.max)
            }
        }
    }
}

/// Generates synthetic prompts with exact token counts.
pub struct SyntheticDataGenerator {
    prompt_dist: TokenDistribution,
    turn_prompt_tokens: Option<usize>,
    tokenizer: Arc<Tokenizer>,
    seed: u64,
    common_prefix_ratio: f64,
    common_prefix_text: Option<String>,
}

impl SyntheticDataGenerator {
    /// Create a new synthetic data generator.
    pub fn new(config: &SyntheticConfig, tokenizer: Arc<Tokenizer>, seed: u64) -> Self {
        // Generate common prefix text if needed
        let common_prefix_text = if config.common_prefix_tokens > 0 {
            Some(generate_fixed_text(
                config.common_prefix_tokens,
                tokenizer.clone(),
                seed,
            ))
        } else {
            None
        };

        Self {
            prompt_dist: TokenDistribution::new(
                config.prompt_tokens,
                config.prompt_tokens_stdev,
                config.prompt_tokens_min,
                config.prompt_tokens_max,
                seed,
            ),
            turn_prompt_tokens: config.turn_prompt_tokens,
            tokenizer,
            seed,
            common_prefix_ratio: config.common_prefix_sample_ratio,
            common_prefix_text,
        }
    }

    /// Generate a single workload with sampled token counts.
    pub fn generate_workload(
        &mut self,
        index: usize,
        max_tokens: Option<u32>,
        turns: usize,
    ) -> Workload {
        // Determine turn_prompt_tokens for each turn
        let first_turn_tokens = self.prompt_dist.sample();
        let subsequent_turn_tokens: Vec<usize> = (0..turns.saturating_sub(1))
            .map(|_| self.prompt_dist.sample())
            .collect();

        let mut user_turns = Vec::with_capacity(turns);

        // First turn: use common_prefix logic
        let prompt_tokens = first_turn_tokens;
        // Deterministic strided selection: for ratio=0.5 this picks indices 0,2,4,…
        // (i.e. every 1/ratio requests). Not random — predictable for reproducible cache tests.
        let use_common_prefix = if self.common_prefix_ratio > 0.0 {
            let ratio_index = (index as f64) / (1.0 / self.common_prefix_ratio);
            ratio_index.fract() < self.common_prefix_ratio
        } else {
            false
        };
        user_turns.push(self.generate_prompt_for_turn(
            prompt_tokens,
            index,
            0,
            use_common_prefix,
            max_tokens,
        ));

        // Subsequent turns: no common prefix, each uses its own token count
        for (turn_idx, &turn_token_count) in subsequent_turn_tokens.iter().enumerate() {
            let token_count = if let Some(tp) = self.turn_prompt_tokens {
                tp
            } else {
                turn_token_count
            };
            user_turns.push(self.generate_prompt_for_turn(
                token_count,
                index,
                turn_idx + 1,
                false,
                max_tokens,
            ));
        }

        if user_turns.len() == 1 {
            Workload::SingleTurn(Prompt {
                prompt: user_turns.pop().unwrap(),
                max_tokens,
            })
        } else {
            Workload::MultiTurn(Conversation {
                system_prompt: None,
                user_turns,
                max_tokens,
            })
        }
    }

    /// Generate prompt text for a specific turn with deterministic prefix.
    fn generate_prompt_for_turn(
        &self,
        token_count: usize,
        index: usize,
        turn_idx: usize,
        use_common_prefix: bool,
        _max_tokens: Option<u32>,
    ) -> String {
        // Use common prefix text if applicable, otherwise empty prefix
        let prefix = if use_common_prefix {
            if let Some(ref common_prefix) = self.common_prefix_text {
                if !common_prefix.is_empty() {
                    common_prefix.clone()
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Suppress unused variable warnings for index and turn_idx
        let _ = (index, turn_idx);

        // Calculate how many tokens we need after the prefix
        let prefix_tokens = self.tokenizer.count_tokens(&prefix);
        let remaining_tokens = if prefix_tokens >= token_count {
            return self.tokenizer.truncate_to_tokens(&prefix, token_count);
        } else {
            token_count - prefix_tokens
        };

        const AVG_CHARS_PER_TOKEN: usize = 5;
        const MARGIN_OF_SAFETY: f64 = 1.5;
        const MAX_ATTEMPTS: usize = 3;

        let mut attempts = 0;
        loop {
            attempts += 1;
            let num_chars = ((remaining_tokens * AVG_CHARS_PER_TOKEN) as f64
                * MARGIN_OF_SAFETY
                * attempts as f64) as usize;

            let mut rng = StdRng::seed_from_u64(self.seed + (index * 1000 + turn_idx) as u64);
            let mut text = String::new();
            while text.len() < num_chars {
                let sentence: String = Sentence(5..20).fake_with_rng(&mut rng);
                text.push_str(&sentence);
                text.push(' ');
            }
            let text = text[..num_chars.min(text.len())].to_string();
            let full_text = format!("{}{}", prefix, text);
            let token_count_actual = self.tokenizer.count_tokens(&full_text);

            if token_count_actual >= token_count {
                return self.tokenizer.truncate_to_tokens(&full_text, token_count);
            }
            if attempts >= MAX_ATTEMPTS {
                warn!(
                    "Failed to generate {} tokens after {} attempts (got {}), using what we have",
                    token_count, MAX_ATTEMPTS, token_count_actual
                );
                return full_text;
            }
        }
    }
}

/// Generate fixed-seed synthetic text of exactly `token_count` tokens.
/// Used for common prefix generation and cache control testing.
///
/// # Arguments
/// * `token_count` - Number of tokens to generate
/// * `tokenizer` - Tokenizer for counting and truncating tokens
/// * `seed` - Random seed for reproducibility
pub fn generate_fixed_text(token_count: usize, tokenizer: Arc<Tokenizer>, seed: u64) -> String {
    const AVG_CHARS_PER_TOKEN: usize = 5;
    const MARGIN_OF_SAFETY: f64 = 1.5;
    const MAX_ATTEMPTS: usize = 3;

    let mut attempts = 0;

    loop {
        attempts += 1;

        // Estimate characters needed
        let num_chars = ((token_count * AVG_CHARS_PER_TOKEN) as f64
            * MARGIN_OF_SAFETY
            * attempts as f64) as usize;

        // Generate random text using fake-rs
        // Use a fixed seed for deterministic output
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate text by concatenating sentences until we have enough characters
        let mut text = String::new();
        while text.len() < num_chars {
            let sentence: String = Sentence(5..20).fake_with_rng(&mut rng);
            text.push_str(&sentence);
            text.push(' ');
        }

        // Truncate to requested length
        let text = text[..num_chars.min(text.len())].to_string();

        // Tokenize
        let token_count_actual = tokenizer.count_tokens(&text);

        if token_count_actual >= token_count {
            return tokenizer.truncate_to_tokens(&text, token_count);
        }

        if attempts >= MAX_ATTEMPTS {
            warn!(
                "Failed to generate fixed text with {} tokens after {} attempts (got {}), using what we have",
                token_count, MAX_ATTEMPTS, token_count_actual
            );
            return text;
        }
    }
}

/// Generate synthetic workloads for benchmarking.
///
/// # Arguments
/// * `config` - Synthetic data configuration
/// * `tokenizer` - Tokenizer for counting tokens
/// * `sample_size` - Number of workloads to generate
/// * `seed` - Random seed for reproducibility
/// * `max_tokens` - Maximum tokens to generate per request (from endpoint.max_tokens)
///
/// # Returns
/// Vector of generated workloads
pub fn generate_synthetic_workloads(
    config: &SyntheticConfig,
    tokenizer: Arc<Tokenizer>,
    sample_size: usize,
    seed: u64,
    max_tokens: Option<u32>,
) -> Result<Vec<Workload>> {
    let turns = config.turns.max(1);
    let mut generator = SyntheticDataGenerator::new(config, tokenizer, seed);

    let workloads: Vec<Workload> = (0..sample_size)
        .map(|i| generator.generate_workload(i, max_tokens, turns))
        .collect();

    Ok(workloads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_distribution_fixed() {
        let mut dist = TokenDistribution::new(100, None, None, None, 42);
        for _ in 0..10 {
            let sample = dist.sample();
            assert_eq!(
                sample, 100,
                "Fixed distribution should always return average"
            );
        }
    }

    #[test]
    fn test_token_distribution_with_bounds() {
        let mut dist = TokenDistribution::new(100, None, Some(50), Some(150), 42);
        for _ in 0..100 {
            let sample = dist.sample();
            assert!(
                (50..=150).contains(&sample),
                "Sample {} outside bounds [50, 150]",
                sample
            );
        }
    }

    #[test]
    fn test_token_distribution_gaussian() {
        let mut dist = TokenDistribution::new(100, Some(20), Some(50), Some(150), 42);
        let mut samples = Vec::new();

        for _ in 0..1000 {
            let sample = dist.sample();
            assert!(
                (50..=150).contains(&sample),
                "Sample {} outside bounds [50, 150]",
                sample
            );
            samples.push(sample);
        }

        // Check that we get some variety (not all the same value)
        let min_sample = *samples.iter().min().unwrap();
        let max_sample = *samples.iter().max().unwrap();
        assert!(
            max_sample > min_sample,
            "Gaussian distribution should produce varied samples"
        );
    }

    #[test]
    fn test_token_distribution_zero_stdev() {
        let mut dist = TokenDistribution::new(100, Some(0), None, None, 42);
        for _ in 0..10 {
            let sample = dist.sample();
            assert_eq!(
                sample, 100,
                "Zero stdev distribution should always return average"
            );
        }
    }

    #[test]
    fn test_generate_prompt_basic() {
        let tokenizer =
            Arc::new(Tokenizer::new("gpt-3.5-turbo").expect("Failed to create tokenizer"));
        let config = SyntheticConfig {
            prompt_tokens: 50,
            prompt_tokens_stdev: None,
            prompt_tokens_min: None,
            prompt_tokens_max: None,
            common_prefix_sample_ratio: 0.0,
            common_prefix_tokens: 0,
            turns: 1,
            turn_prompt_tokens: None,
        };

        let generator = SyntheticDataGenerator::new(&config, tokenizer.clone(), 42);
        let text = generator.generate_prompt_for_turn(50, 0, 0, false, None);

        // Verify token count is close to target (allow some tolerance due to tokenization)
        let token_count = tokenizer.count_tokens(&text);
        assert!(
            (48..=52).contains(&token_count),
            "Token count {} should be close to target 50",
            token_count
        );
    }

    #[test]
    fn test_generate_workload() {
        let tokenizer =
            Arc::new(Tokenizer::new("gpt-3.5-turbo").expect("Failed to create tokenizer"));
        let config = SyntheticConfig {
            prompt_tokens: 100,
            prompt_tokens_stdev: None,
            prompt_tokens_min: None,
            prompt_tokens_max: None,
            common_prefix_sample_ratio: 0.0,
            common_prefix_tokens: 0,
            turns: 1,
            turn_prompt_tokens: None,
        };

        let mut generator = SyntheticDataGenerator::new(&config, tokenizer, 42);
        let workload = generator.generate_workload(0, Some(50), 1);

        match workload {
            Workload::SingleTurn(prompt) => {
                assert_eq!(
                    prompt.max_tokens,
                    Some(50),
                    "Workload should have correct max_tokens"
                );
            }
            _ => panic!("Expected SingleTurn workload"),
        }
    }

    #[test]
    fn test_deterministic_generation() {
        let tokenizer =
            Arc::new(Tokenizer::new("gpt-3.5-turbo").expect("Failed to create tokenizer"));
        let config = SyntheticConfig {
            prompt_tokens: 50,
            prompt_tokens_stdev: None,
            prompt_tokens_min: None,
            prompt_tokens_max: None,
            common_prefix_sample_ratio: 0.0,
            common_prefix_tokens: 0,
            turns: 1,
            turn_prompt_tokens: None,
        };

        // Generate twice with same seed
        let workloads1 = generate_synthetic_workloads(&config, tokenizer.clone(), 10, 42, Some(20))
            .expect("Failed to generate workloads");
        let workloads2 = generate_synthetic_workloads(&config, tokenizer.clone(), 10, 42, Some(20))
            .expect("Failed to generate workloads");

        // Should be identical
        assert_eq!(workloads1.len(), workloads2.len());
        for (w1, w2) in workloads1.iter().zip(workloads2.iter()) {
            match (w1, w2) {
                (Workload::SingleTurn(p1), Workload::SingleTurn(p2)) => {
                    assert_eq!(
                        p1.prompt, p2.prompt,
                        "Same seed should produce identical prompts"
                    );
                    assert_eq!(p1.max_tokens, p2.max_tokens);
                }
                _ => panic!("Expected SingleTurn workloads"),
            }
        }
    }

    #[test]
    fn test_generate_prompt_no_prefix() {
        let tokenizer =
            Arc::new(Tokenizer::new("gpt-3.5-turbo").expect("Failed to create tokenizer"));
        let config = SyntheticConfig {
            prompt_tokens: 50,
            prompt_tokens_stdev: None,
            prompt_tokens_min: None,
            prompt_tokens_max: None,
            common_prefix_sample_ratio: 0.0,
            common_prefix_tokens: 0,
            turns: 1,
            turn_prompt_tokens: None,
        };

        let generator = SyntheticDataGenerator::new(&config, tokenizer.clone(), 42);
        let text = generator.generate_prompt_for_turn(50, 0, 0, false, None);

        // add_prefix has been removed; prompts should not contain synthetic prefix markers
        assert!(
            !text.starts_with("[synthetic-"),
            "Prompt should not have synthetic prefix after add_prefix removal"
        );

        // Verify token count is still close to target
        let token_count = tokenizer.count_tokens(&text);
        assert!(
            (48..=52).contains(&token_count),
            "Token count {} should be close to target 50",
            token_count
        );
    }

    #[test]
    fn test_common_prefix_generation() {
        let tokenizer =
            Arc::new(Tokenizer::new("gpt-3.5-turbo").expect("Failed to create tokenizer"));
        let config = SyntheticConfig {
            prompt_tokens: 100,
            prompt_tokens_stdev: None,
            prompt_tokens_min: None,
            prompt_tokens_max: None,
            common_prefix_sample_ratio: 0.5, // 50% should share common prefix
            common_prefix_tokens: 50,
            turns: 1,
            turn_prompt_tokens: None,
        };

        let mut generator = SyntheticDataGenerator::new(&config, tokenizer.clone(), 42);

        // Generate multiple workloads
        let workloads: Vec<_> = (0..10)
            .map(|i| generator.generate_workload(i, Some(50), 1))
            .collect();

        // Extract prompts
        let prompts: Vec<String> = workloads
            .iter()
            .map(|w| match w {
                Workload::SingleTurn(p) => p.prompt.clone(),
                _ => panic!("Expected SingleTurn workload"),
            })
            .collect();

        // Check that some prompts share a common prefix
        // With 50% ratio, approximately half should start with the same text
        let common_prefix_text = generator.common_prefix_text.as_ref().unwrap();
        let with_common_prefix: Vec<_> = prompts
            .iter()
            .filter(|p| p.starts_with(common_prefix_text.as_str()))
            .collect();

        // Should have some prompts with common prefix
        assert!(
            !with_common_prefix.is_empty(),
            "Some prompts should have common prefix"
        );

        // Verify token count is correct for all prompts
        for prompt in &prompts {
            let token_count = tokenizer.count_tokens(prompt);
            assert!(
                (98..=102).contains(&token_count),
                "Token count {} should be close to target 100",
                token_count
            );
        }
    }

    #[test]
    fn test_multi_turn_generation() {
        let tokenizer =
            Arc::new(Tokenizer::new("gpt-3.5-turbo").expect("Failed to create tokenizer"));
        let config = SyntheticConfig {
            prompt_tokens: 64,
            prompt_tokens_stdev: None,
            prompt_tokens_min: None,
            prompt_tokens_max: None,
            common_prefix_sample_ratio: 0.0,
            common_prefix_tokens: 0,
            turns: 3,
            turn_prompt_tokens: Some(48),
        };

        let mut generator = SyntheticDataGenerator::new(&config, tokenizer.clone(), 42);
        let workload = generator.generate_workload(0, Some(50), 3);

        match workload {
            Workload::MultiTurn(conv) => {
                assert_eq!(conv.user_turns.len(), 3, "Should have 3 user turns");
                assert_eq!(conv.max_tokens, Some(50));

                // First turn uses prompt_tokens (64)
                let first_tokens = tokenizer.count_tokens(&conv.user_turns[0]);
                assert!(
                    (62..=68).contains(&first_tokens),
                    "First turn token count {} should be close to 64",
                    first_tokens
                );

                // Subsequent turns use turn_prompt_tokens (48)
                for (i, turn) in conv.user_turns.iter().skip(1).enumerate() {
                    let turn_idx = i + 1;
                    let turn_tokens = tokenizer.count_tokens(turn);
                    assert!(
                        (46..=52).contains(&turn_tokens),
                        "Turn {} token count {} should be close to 48",
                        turn_idx,
                        turn_tokens
                    );
                }
            }
            _ => panic!("Expected MultiTurn workload"),
        }
    }

    #[test]
    fn test_multi_turn_deterministic() {
        let tokenizer =
            Arc::new(Tokenizer::new("gpt-3.5-turbo").expect("Failed to create tokenizer"));
        let config = SyntheticConfig {
            prompt_tokens: 50,
            prompt_tokens_stdev: None,
            prompt_tokens_min: None,
            prompt_tokens_max: None,
            common_prefix_sample_ratio: 0.0,
            common_prefix_tokens: 0,
            turns: 4,
            turn_prompt_tokens: None,
        };

        let workloads1 = generate_synthetic_workloads(&config, tokenizer.clone(), 5, 42, Some(100))
            .expect("Failed to generate workloads");
        let workloads2 = generate_synthetic_workloads(&config, tokenizer.clone(), 5, 42, Some(100))
            .expect("Failed to generate workloads");

        for (w1, w2) in workloads1.iter().zip(workloads2.iter()) {
            match (w1, w2) {
                (Workload::MultiTurn(c1), Workload::MultiTurn(c2)) => {
                    assert_eq!(c1.user_turns.len(), c2.user_turns.len());
                    for (t1, t2) in c1.user_turns.iter().zip(c2.user_turns.iter()) {
                        assert_eq!(t1, t2, "Same seed should produce identical turns");
                    }
                }
                _ => panic!("Expected MultiTurn workloads"),
            }
        }
    }
}

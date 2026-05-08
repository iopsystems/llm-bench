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

use crate::benchmark::{Prompt, Workload};
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
    tokenizer: Arc<Tokenizer>,
    seed: u64,
    add_prefix: bool,
}

impl SyntheticDataGenerator {
    /// Create a new synthetic data generator.
    pub fn new(config: &SyntheticConfig, tokenizer: Arc<Tokenizer>, seed: u64) -> Self {
        Self {
            prompt_dist: TokenDistribution::new(
                config.prompt_tokens,
                config.prompt_tokens_stdev,
                config.prompt_tokens_min,
                config.prompt_tokens_max,
                seed,
            ),
            tokenizer,
            seed,
            add_prefix: config.add_prefix,
        }
    }

    /// Generate a single workload with sampled token counts.
    pub fn generate_workload(&mut self, index: usize, max_tokens: Option<u32>) -> Workload {
        let prompt_tokens = self.prompt_dist.sample();

        let prompt_text = self.generate_prompt(prompt_tokens, index);

        Workload::SingleTurn(Prompt {
            prompt: prompt_text,
            max_tokens,
        })
    }

    /// Generate prompt text with exact token count.
    ///
    /// Uses guidellm's algorithm:
    /// 1. Estimate chars needed: tokens × 5 × 1.5 × attempts
    /// 2. Generate random text with fake-rs
    /// 3. Tokenize and check count
    /// 4. If not enough tokens, retry with more chars
    /// 5. Truncate to exact token count
    fn generate_prompt(&self, token_count: usize, index: usize) -> String {
        // Add unique prefix for cache busting (if enabled)
        let prefix = if self.add_prefix {
            format!("[synthetic-{}] ", index)
        } else {
            String::new()
        };

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
            // Use a deterministic seed based on index for reproducibility
            let mut rng = StdRng::seed_from_u64(self.seed + index as u64);

            // Generate text by concatenating sentences until we have enough characters
            let mut text = String::new();
            while text.len() < num_chars {
                let sentence: String = Sentence(5..20).fake_with_rng(&mut rng);
                text.push_str(&sentence);
                text.push(' ');
            }

            // Truncate to requested length
            let text = text[..num_chars.min(text.len())].to_string();

            let full_text = format!("{}{}", prefix, text);

            // Tokenize
            let token_count_actual = self.tokenizer.count_tokens(&full_text);

            if token_count_actual >= token_count {
                // Success - we have enough tokens
                // Now truncate to exact count by encoding/decoding
                return full_text[..self.truncate_to_tokens(&full_text, token_count)].to_string();
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

    /// Truncate text to exact token count by finding the character boundary.
    fn truncate_to_tokens(&self, text: &str, target_tokens: usize) -> usize {
        // Binary search to find the right character position
        let mut low = 0;
        let mut high = text.len();

        while low < high {
            let mid = (low + high).div_ceil(2);
            let tokens = self.tokenizer.count_tokens(&text[..mid]);

            if tokens <= target_tokens {
                low = mid;
            } else {
                high = mid - 1;
            }
        }

        low
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
    let mut generator = SyntheticDataGenerator::new(config, tokenizer, seed);

    let workloads: Vec<Workload> = (0..sample_size)
        .map(|i| generator.generate_workload(i, max_tokens))
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
            add_prefix: true,
        };

        let generator = SyntheticDataGenerator::new(&config, tokenizer.clone(), 42);
        let text = generator.generate_prompt(50, 0);

        // Verify it has the prefix
        assert!(
            text.starts_with("[synthetic-0]"),
            "Prompt should have cache-busting prefix"
        );

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
            add_prefix: true,
        };

        let mut generator = SyntheticDataGenerator::new(&config, tokenizer, 42);
        let workload = generator.generate_workload(0, Some(50));

        match workload {
            Workload::SingleTurn(prompt) => {
                assert!(
                    prompt.prompt.starts_with("[synthetic-0]"),
                    "Workload should have cache-busting prefix"
                );
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
            add_prefix: true,
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
            add_prefix: false,
        };

        let generator = SyntheticDataGenerator::new(&config, tokenizer.clone(), 42);
        let text = generator.generate_prompt(50, 0);

        // Verify it does NOT have the prefix
        assert!(
            !text.starts_with("[synthetic-"),
            "Prompt should not have prefix when add_prefix=false"
        );

        // Verify token count is still close to target
        let token_count = tokenizer.count_tokens(&text);
        assert!(
            (48..=52).contains(&token_count),
            "Token count {} should be close to target 50",
            token_count
        );
    }
}

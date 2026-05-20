use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::thread_rng;
use rand_distr::{Distribution, Exp, Normal};
use std::time::Duration;

use crate::config::{ArrivalDistribution, ConversationConfig};

/// Manages request arrival patterns for load testing
pub struct RequestDistribution {
    dist_type: DistributionType,
}

enum DistributionType {
    Uniform { interval: Duration },
    Exponential { exp_dist: Exp<f64> },
}

impl RequestDistribution {
    /// Create a new request distribution
    ///
    /// # Arguments
    /// * `arrival_dist` - Type of distribution (uniform or poisson)
    /// * `qps` - Target queries per second
    pub fn new(arrival_dist: &ArrivalDistribution, qps: f64) -> Self {
        let dist_type = match arrival_dist {
            ArrivalDistribution::Uniform => {
                let interval_ms = (1000.0 / qps) as u64;
                DistributionType::Uniform {
                    interval: Duration::from_millis(interval_ms),
                }
            }
            ArrivalDistribution::Poisson => {
                // For Poisson arrivals, inter-arrival times follow exponential distribution
                // λ (lambda) = rate = qps
                let exp_dist =
                    Exp::new(qps).expect("QPS must be positive for exponential distribution");
                DistributionType::Exponential { exp_dist }
            }
        };

        Self { dist_type }
    }

    /// Get the next delay duration before sending a request
    ///
    /// For uniform distribution, returns a fixed interval.
    /// For Poisson/exponential, samples from the distribution.
    pub fn next_delay(&self) -> Duration {
        match &self.dist_type {
            DistributionType::Uniform { interval } => *interval,
            DistributionType::Exponential { exp_dist } => {
                let mut rng = thread_rng();
                // Sample returns time in seconds
                let wait_secs = exp_dist.sample(&mut rng);
                Duration::from_secs_f64(wait_secs)
            }
        }
    }

    /// Get the distribution type as a string for logging
    pub fn distribution_name(&self) -> &str {
        match &self.dist_type {
            DistributionType::Uniform { .. } => "Uniform",
            DistributionType::Exponential { .. } => "Poisson",
        }
    }
}

/// Per-conversation sampler for inter-turn delays.
///
/// Samples from a Gaussian centered at `mean_ms` with `stdev_ms` standard deviation,
/// clamped to `[min_ms, max_ms]`. With `stdev_ms == 0`, the sampler returns `mean_ms`
/// exactly (clamped) and consumes no entropy from the RNG.
pub struct TurnDelayDistribution {
    mean_ms: u64,
    stdev_ms: u64,
    min_ms: u64,
    max_ms: u64,
    rng: StdRng,
}

impl TurnDelayDistribution {
    pub fn new(config: &ConversationConfig, seed: u64) -> Self {
        Self {
            mean_ms: config.turn_delay_ms,
            stdev_ms: config.turn_delay_stdev_ms,
            min_ms: config.turn_delay_min_ms,
            max_ms: config.turn_delay_max_ms,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Returns true if every sample would be exactly zero — the caller can skip
    /// the entire delay-sampling-and-sleeping machinery without changing observable
    /// behavior.
    pub fn is_no_op(config: &ConversationConfig) -> bool {
        config.turn_delay_ms == 0
            && config.turn_delay_stdev_ms == 0
            && config.turn_delay_min_ms == 0
    }

    /// Sample the next inter-turn delay.
    pub fn sample(&mut self) -> Duration {
        let ms = if self.stdev_ms == 0 {
            self.mean_ms.clamp(self.min_ms, self.max_ms)
        } else {
            let normal = Normal::new(self.mean_ms as f64, self.stdev_ms as f64)
                .expect("non-negative stdev produces valid Normal distribution");
            let sample = normal.sample(&mut self.rng).round();
            // Clamp in f64 space before casting so negative draws floor to min_ms,
            // not whatever `as u64` decides to do with a negative float.
            let lo = self.min_ms as f64;
            let hi = self.max_ms as f64;
            sample.clamp(lo, hi) as u64
        };
        Duration::from_millis(ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        let dist = RequestDistribution::new(&ArrivalDistribution::Uniform, 10.0);

        // For 10 QPS, expect 100ms intervals
        let delay = dist.next_delay();
        assert_eq!(delay, Duration::from_millis(100));

        // Should be deterministic
        let delay2 = dist.next_delay();
        assert_eq!(delay, delay2);
    }

    #[test]
    fn test_poisson_distribution_variability() {
        let dist = RequestDistribution::new(&ArrivalDistribution::Poisson, 10.0);

        // Collect several samples
        let mut delays = Vec::new();
        for _ in 0..100 {
            delays.push(dist.next_delay());
        }

        // Poisson should produce variable delays
        let all_same = delays.iter().all(|d| *d == delays[0]);
        assert!(
            !all_same,
            "Poisson distribution should produce variable delays"
        );

        // Average should be roughly 1/rate = 1/10 = 0.1 seconds
        let avg_secs: f64 =
            delays.iter().map(|d| d.as_secs_f64()).sum::<f64>() / delays.len() as f64;
        assert!(
            (avg_secs - 0.1).abs() < 0.05,
            "Average delay should be close to 1/rate (0.1s), got {}",
            avg_secs
        );
    }

    #[test]
    fn test_distribution_name() {
        let uniform = RequestDistribution::new(&ArrivalDistribution::Uniform, 10.0);
        assert_eq!(uniform.distribution_name(), "Uniform");

        let poisson = RequestDistribution::new(&ArrivalDistribution::Poisson, 10.0);
        assert_eq!(poisson.distribution_name(), "Poisson");
    }

    #[test]
    fn turn_delay_returns_mean_when_stdev_zero() {
        let cfg = ConversationConfig {
            turn_delay_ms: 250,
            turn_delay_stdev_ms: 0,
            turn_delay_min_ms: 0,
            turn_delay_max_ms: 60_000,
        };
        let mut dist = TurnDelayDistribution::new(&cfg, 1);
        for _ in 0..5 {
            assert_eq!(dist.sample(), Duration::from_millis(250));
        }
    }

    #[test]
    fn turn_delay_clamps_to_min_max() {
        // Force the mean way above max so every sample clamps.
        let cfg = ConversationConfig {
            turn_delay_ms: 10_000,
            turn_delay_stdev_ms: 5_000,
            turn_delay_min_ms: 100,
            turn_delay_max_ms: 200,
        };
        let mut dist = TurnDelayDistribution::new(&cfg, 42);
        for _ in 0..100 {
            let d = dist.sample().as_millis() as u64;
            assert!((100..=200).contains(&d), "sample {} out of bounds", d);
        }
    }

    #[test]
    fn turn_delay_is_deterministic_for_seed() {
        let cfg = ConversationConfig {
            turn_delay_ms: 500,
            turn_delay_stdev_ms: 100,
            turn_delay_min_ms: 0,
            turn_delay_max_ms: 60_000,
        };
        let mut a = TurnDelayDistribution::new(&cfg, 7);
        let mut b = TurnDelayDistribution::new(&cfg, 7);
        for _ in 0..20 {
            assert_eq!(a.sample(), b.sample());
        }
    }

    #[test]
    fn turn_delay_is_no_op_for_defaults() {
        assert!(TurnDelayDistribution::is_no_op(
            &ConversationConfig::default()
        ));
        assert!(!TurnDelayDistribution::is_no_op(&ConversationConfig {
            turn_delay_ms: 1,
            ..ConversationConfig::default()
        }));
        assert!(!TurnDelayDistribution::is_no_op(&ConversationConfig {
            turn_delay_stdev_ms: 1,
            ..ConversationConfig::default()
        }));
    }

    #[test]
    fn turn_delay_three_turn_total_matches_spec_example() {
        // Spec test #2: a 3-turn conversation with turn_delay_ms=200, stdev=0
        // has two gaps and totals 400 ms of inter-turn delay.
        let cfg = ConversationConfig {
            turn_delay_ms: 200,
            turn_delay_stdev_ms: 0,
            turn_delay_min_ms: 0,
            turn_delay_max_ms: 60_000,
        };
        let mut dist = TurnDelayDistribution::new(&cfg, 0);
        let total: Duration = (0..2).map(|_| dist.sample()).sum();
        assert_eq!(total, Duration::from_millis(400));
    }

    #[test]
    fn turn_delay_zero_mean_with_stdev_stays_non_negative() {
        let cfg = ConversationConfig {
            turn_delay_ms: 0,
            turn_delay_stdev_ms: 1000,
            turn_delay_min_ms: 0,
            turn_delay_max_ms: 60_000,
        };
        let mut dist = TurnDelayDistribution::new(&cfg, 11);
        for _ in 0..200 {
            let d = dist.sample();
            assert!(d.as_millis() <= 60_000);
        }
    }
}

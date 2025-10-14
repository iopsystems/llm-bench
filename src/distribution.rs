use rand::thread_rng;
use rand_distr::{Distribution, Exp};
use std::time::Duration;

use crate::config::ArrivalDistribution;

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
}

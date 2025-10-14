use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    // Metadata
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
    pub version: String,

    // Test configuration
    pub configuration: TestConfiguration,

    // Results
    pub summary: Summary,
    pub throughput: ThroughputStats,
    pub latency: LatencyStats,
    pub errors: ErrorBreakdown,

    // Context-aware metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_latency: Option<ContextLatencyStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_itl: Option<ContextITLStats>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TestConfiguration {
    pub endpoint: String,
    pub model: String,
    pub load_pattern: String,
    pub concurrent_requests: Option<usize>,
    pub qps: Option<f64>,
    pub total_requests: Option<usize>,
    pub duration_seconds: Option<u64>,
    pub warmup_requests: Option<usize>,
    pub warmup_duration: Option<u64>,
    pub prompt_file: String,
    pub shuffle: bool,
    pub sample_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBreakdown {
    pub timeout_errors: u64,
    pub connection_errors: u64,
    pub http_4xx_errors: u64,
    pub http_5xx_errors: u64,
    pub other_errors: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub success_rate: f64,
    pub retries: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThroughputStats {
    pub requests_per_second: f64,
    pub input_tokens_per_second: f64,
    pub output_tokens_per_second: f64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyStats {
    pub ttft_p50_ms: f64,
    pub ttft_p90_ms: f64,
    pub ttft_p95_ms: f64,
    pub ttft_p99_ms: f64,
    pub itl_p50_ms: f64,
    pub itl_p90_ms: f64,
    pub itl_p95_ms: f64,
    pub itl_p99_ms: f64,
    pub request_p50_ms: f64,
    pub request_p90_ms: f64,
    pub request_p95_ms: f64,
    pub request_p99_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextLatencyStats {
    pub small: LatencyPercentiles,
    pub medium: LatencyPercentiles,
    pub large: LatencyPercentiles,
    pub xlarge: LatencyPercentiles,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xxlarge: Option<LatencyPercentiles>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyPercentiles {
    pub ttft_p50_ms: f64,
    pub ttft_p90_ms: f64,
    pub ttft_p95_ms: f64,
    pub ttft_p99_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextITLStats {
    pub small: ITLPercentiles,
    pub medium: ITLPercentiles,
    pub large: ITLPercentiles,
    pub xlarge: ITLPercentiles,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xxlarge: Option<ITLPercentiles>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ITLPercentiles {
    pub itl_p50_ms: f64,
    pub itl_p90_ms: f64,
    pub itl_p95_ms: f64,
    pub itl_p99_ms: f64,
}

pub struct ReportBuilder {
    start_time: SystemTime,
    config: Option<crate::config::Config>,
}

impl Default for ReportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ReportBuilder {
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            config: None,
        }
    }

    pub fn with_config(mut self, config: crate::config::Config) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build(&self) -> Result<BenchmarkReport> {
        let end_time = SystemTime::now();
        let duration = end_time.duration_since(self.start_time)?;

        // Access metrics directly for now
        // In production, we'd use metriken-exposition properly
        use crate::metrics::{
            ERRORS_CONNECTION, ERRORS_HTTP_4XX, ERRORS_HTTP_5XX, ERRORS_OTHER, REQUESTS_FAILED,
            REQUESTS_RETRIED, REQUESTS_SENT, REQUESTS_SUCCESS, REQUESTS_TIMEOUT, TOKENS_INPUT,
            TOKENS_OUTPUT,
        };

        let requests_sent = REQUESTS_SENT.value();
        let requests_success = REQUESTS_SUCCESS.value();
        let requests_failed = REQUESTS_FAILED.value();
        let requests_timeout = REQUESTS_TIMEOUT.value();

        let input_tokens = TOKENS_INPUT.value();
        let output_tokens = TOKENS_OUTPUT.value();

        let duration_secs = duration.as_secs_f64();

        // Build test configuration from config
        let configuration = if let Some(ref config) = self.config {
            TestConfiguration {
                endpoint: config.endpoint.base_url.clone(),
                model: config
                    .endpoint
                    .model
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                load_pattern: if config.load.qps.is_some() {
                    "FixedQps".to_string()
                } else {
                    "Concurrent".to_string()
                },
                concurrent_requests: Some(config.load.concurrent_requests),
                qps: config.load.qps,
                total_requests: config.load.total_requests,
                duration_seconds: config.load.duration_seconds,
                warmup_requests: config.load.warmup_requests,
                warmup_duration: config.load.warmup_duration,
                prompt_file: config.input.file.display().to_string(),
                shuffle: config.input.shuffle,
                sample_size: config.input.sample_size,
            }
        } else {
            // Default configuration if not provided
            TestConfiguration {
                endpoint: "unknown".to_string(),
                model: "unknown".to_string(),
                load_pattern: "unknown".to_string(),
                concurrent_requests: None,
                qps: None,
                total_requests: None,
                duration_seconds: None,
                warmup_requests: None,
                warmup_duration: None,
                prompt_file: "unknown".to_string(),
                shuffle: false,
                sample_size: None,
            }
        };

        let retries = REQUESTS_RETRIED.value();

        // Calculate completed requests (successful + failed)
        let completed_requests = requests_success + requests_failed + requests_timeout;

        let summary = Summary {
            total_requests: requests_sent,
            successful_requests: requests_success,
            failed_requests: requests_failed + requests_timeout,
            success_rate: if completed_requests > 0 {
                requests_success as f64 / completed_requests as f64
            } else {
                0.0
            },
            retries,
        };

        let throughput = ThroughputStats {
            requests_per_second: requests_sent as f64 / duration_secs,
            input_tokens_per_second: input_tokens as f64 / duration_secs,
            output_tokens_per_second: output_tokens as f64 / duration_secs,
            total_input_tokens: input_tokens,
            total_output_tokens: output_tokens,
        };

        // Build error breakdown
        let errors = ErrorBreakdown {
            timeout_errors: requests_timeout,
            connection_errors: ERRORS_CONNECTION.value(),
            http_4xx_errors: ERRORS_HTTP_4XX.value(),
            http_5xx_errors: ERRORS_HTTP_5XX.value(),
            other_errors: ERRORS_OTHER.value(),
        };

        // Get latency percentiles
        let latency = self.build_latency_stats()?;
        let context_latency = self.build_context_latency_stats();
        let context_itl = self.build_context_itl_stats();

        // Convert SystemTime to DateTime<Utc>
        let timestamp: DateTime<Utc> = self.start_time.into();

        Ok(BenchmarkReport {
            timestamp,
            duration,
            version: env!("CARGO_PKG_VERSION").to_string(),
            configuration,
            summary,
            throughput,
            latency,
            errors,
            context_latency,
            context_itl,
        })
    }

    fn build_latency_stats(&self) -> Result<LatencyStats> {
        use crate::metrics::{INTER_TOKEN_LATENCY, REQUEST_LATENCY, TTFT};

        // Extract TTFT percentiles
        let mut ttft_p50 = 0.0;
        let mut ttft_p90 = 0.0;
        let mut ttft_p95 = 0.0;
        let mut ttft_p99 = 0.0;

        if let Some(ttft_histogram) = TTFT.load()
            && let Ok(Some(percentiles)) = ttft_histogram.percentiles(&[50.0, 90.0, 95.0, 99.0])
        {
            for (percentile, bucket) in percentiles.iter() {
                let value_ms = bucket.end() as f64 / 1_000_000.0;
                match percentile.round() as u32 {
                    50 => ttft_p50 = value_ms,
                    90 => ttft_p90 = value_ms,
                    95 => ttft_p95 = value_ms,
                    99 => ttft_p99 = value_ms,
                    _ => {}
                }
            }
        }

        // Extract ITL percentiles
        let mut itl_p50 = 0.0;
        let mut itl_p90 = 0.0;
        let mut itl_p95 = 0.0;
        let mut itl_p99 = 0.0;

        if let Some(itl_histogram) = INTER_TOKEN_LATENCY.load()
            && let Ok(Some(percentiles)) = itl_histogram.percentiles(&[50.0, 90.0, 95.0, 99.0])
        {
            for (percentile, bucket) in percentiles.iter() {
                let value_ms = bucket.end() as f64 / 1_000_000.0;
                match percentile.round() as u32 {
                    50 => itl_p50 = value_ms,
                    90 => itl_p90 = value_ms,
                    95 => itl_p95 = value_ms,
                    99 => itl_p99 = value_ms,
                    _ => {}
                }
            }
        }

        // Extract request latency percentiles
        let mut request_p50 = 0.0;
        let mut request_p90 = 0.0;
        let mut request_p95 = 0.0;
        let mut request_p99 = 0.0;

        if let Some(request_histogram) = REQUEST_LATENCY.load()
            && let Ok(Some(percentiles)) = request_histogram.percentiles(&[50.0, 90.0, 95.0, 99.0])
        {
            for (percentile, bucket) in percentiles.iter() {
                let value_ms = bucket.end() as f64 / 1_000_000.0;
                match percentile.round() as u32 {
                    50 => request_p50 = value_ms,
                    90 => request_p90 = value_ms,
                    95 => request_p95 = value_ms,
                    99 => request_p99 = value_ms,
                    _ => {}
                }
            }
        }

        Ok(LatencyStats {
            ttft_p50_ms: ttft_p50,
            ttft_p90_ms: ttft_p90,
            ttft_p95_ms: ttft_p95,
            ttft_p99_ms: ttft_p99,
            itl_p50_ms: itl_p50,
            itl_p90_ms: itl_p90,
            itl_p95_ms: itl_p95,
            itl_p99_ms: itl_p99,
            request_p50_ms: request_p50,
            request_p90_ms: request_p90,
            request_p95_ms: request_p95,
            request_p99_ms: request_p99,
        })
    }

    fn build_context_latency_stats(&self) -> Option<ContextLatencyStats> {
        use crate::metrics::{TTFT_LARGE, TTFT_MEDIUM, TTFT_SMALL, TTFT_XLARGE, TTFT_XXLARGE};

        let extract_percentiles =
            |histogram: &metriken::AtomicHistogram| -> Option<LatencyPercentiles> {
                if let Some(loaded) = histogram.load()
                    && let Ok(Some(percentiles)) = loaded.percentiles(&[50.0, 90.0, 95.0, 99.0])
                    && !percentiles.is_empty()
                {
                    let mut p50 = 0.0;
                    let mut p90 = 0.0;
                    let mut p95 = 0.0;
                    let mut p99 = 0.0;

                    for (percentile, bucket) in percentiles.iter() {
                        let value_ms = bucket.end() as f64 / 1_000_000.0;
                        match percentile.round() as u32 {
                            50 => p50 = value_ms,
                            90 => p90 = value_ms,
                            95 => p95 = value_ms,
                            99 => p99 = value_ms,
                            _ => {}
                        }
                    }

                    return Some(LatencyPercentiles {
                        ttft_p50_ms: p50,
                        ttft_p90_ms: p90,
                        ttft_p95_ms: p95,
                        ttft_p99_ms: p99,
                    });
                }
                None
            };

        // Check if we have any context data
        let has_data = TTFT_SMALL.load().is_some()
            || TTFT_MEDIUM.load().is_some()
            || TTFT_LARGE.load().is_some()
            || TTFT_XLARGE.load().is_some()
            || TTFT_XXLARGE.load().is_some();

        if !has_data {
            return None;
        }

        // Extract percentiles for each bucket
        let small = extract_percentiles(&TTFT_SMALL)?;
        let medium = extract_percentiles(&TTFT_MEDIUM)?;
        let large = extract_percentiles(&TTFT_LARGE)?;
        let xlarge = extract_percentiles(&TTFT_XLARGE)?;
        let xxlarge = extract_percentiles(&TTFT_XXLARGE);

        Some(ContextLatencyStats {
            small,
            medium,
            large,
            xlarge,
            xxlarge,
        })
    }

    fn build_context_itl_stats(&self) -> Option<ContextITLStats> {
        use crate::metrics::{ITL_LARGE, ITL_MEDIUM, ITL_SMALL, ITL_XLARGE, ITL_XXLARGE};

        let extract_percentiles =
            |histogram: &metriken::AtomicHistogram| -> Option<ITLPercentiles> {
                if let Some(loaded) = histogram.load()
                    && let Ok(Some(percentiles)) = loaded.percentiles(&[50.0, 90.0, 95.0, 99.0])
                    && !percentiles.is_empty()
                {
                    let mut p50 = 0.0;
                    let mut p90 = 0.0;
                    let mut p95 = 0.0;
                    let mut p99 = 0.0;

                    for (percentile, bucket) in percentiles.iter() {
                        let value_ms = bucket.end() as f64 / 1_000_000.0;
                        match percentile.round() as u32 {
                            50 => p50 = value_ms,
                            90 => p90 = value_ms,
                            95 => p95 = value_ms,
                            99 => p99 = value_ms,
                            _ => {}
                        }
                    }

                    return Some(ITLPercentiles {
                        itl_p50_ms: p50,
                        itl_p90_ms: p90,
                        itl_p95_ms: p95,
                        itl_p99_ms: p99,
                    });
                }
                None
            };

        // Check if we have any context ITL data
        let has_data = ITL_SMALL.load().is_some()
            || ITL_MEDIUM.load().is_some()
            || ITL_LARGE.load().is_some()
            || ITL_XLARGE.load().is_some()
            || ITL_XXLARGE.load().is_some();

        if !has_data {
            return None;
        }

        // Extract percentiles for each bucket
        let small = extract_percentiles(&ITL_SMALL)?;
        let medium = extract_percentiles(&ITL_MEDIUM)?;
        let large = extract_percentiles(&ITL_LARGE)?;
        let xlarge = extract_percentiles(&ITL_XLARGE)?;
        let xxlarge = extract_percentiles(&ITL_XXLARGE);

        Some(ContextITLStats {
            small,
            medium,
            large,
            xlarge,
            xxlarge,
        })
    }

    pub fn print_console_report(&self) -> Result<()> {
        let report = self.build()?;

        use chrono::Utc;
        let now = Utc::now();
        let timestamp = now.to_rfc3339_opts(chrono::SecondsFormat::Millis, false);

        println!();
        println!("{}", timestamp);
        println!("{} -----", timestamp);
        println!("{} Benchmark Complete", timestamp);
        println!(
            "{} Duration: {:.1}s",
            timestamp,
            report.duration.as_secs_f64()
        );
        println!(
            "{} Requests: Sent: {} Retries: {}",
            timestamp, report.summary.total_requests, report.summary.retries
        );
        println!(
            "{} Responses: Received: {} Ok: {} Err: {} Success: {:.2}%",
            timestamp,
            report.summary.successful_requests + report.summary.failed_requests,
            report.summary.successful_requests,
            report.summary.failed_requests,
            report.summary.success_rate * 100.0
        );

        // Error breakdown if any
        let total_errors = report.errors.timeout_errors
            + report.errors.connection_errors
            + report.errors.http_4xx_errors
            + report.errors.http_5xx_errors
            + report.errors.other_errors;
        if total_errors > 0 {
            println!(
                "{} Errors: Connection: {} 4xx: {} 5xx: {} Timeout: {} Other: {}",
                timestamp,
                report.errors.connection_errors,
                report.errors.http_4xx_errors,
                report.errors.http_5xx_errors,
                report.errors.timeout_errors,
                report.errors.other_errors
            );
        }

        println!(
            "{} Tokens: Input: {} Output: {} Total: {}",
            timestamp,
            report.throughput.total_input_tokens,
            report.throughput.total_output_tokens,
            report.throughput.total_input_tokens + report.throughput.total_output_tokens
        );

        println!(
            "{} Throughput: Requests/s: {:.2} Input tokens/s: {:.2} Output tokens/s: {:.2}",
            timestamp,
            report.throughput.requests_per_second,
            report.throughput.input_tokens_per_second,
            report.throughput.output_tokens_per_second
        );

        println!(
            "{} TTFT (ms): p50: {:.0} p90: {:.0} p99: {:.0}",
            timestamp,
            report.latency.ttft_p50_ms,
            report.latency.ttft_p90_ms,
            report.latency.ttft_p99_ms
        );

        if report.latency.itl_p50_ms > 0.0 {
            println!(
                "{} ITL (ms): p50: {:.0} p90: {:.0} p99: {:.0}",
                timestamp,
                report.latency.itl_p50_ms,
                report.latency.itl_p90_ms,
                report.latency.itl_p99_ms
            );
        }

        println!(
            "{} Request Latency (ms): p50: {:.0} p90: {:.0} p99: {:.0}",
            timestamp,
            report.latency.request_p50_ms,
            report.latency.request_p90_ms,
            report.latency.request_p99_ms
        );

        println!("\n");

        Ok(())
    }
}

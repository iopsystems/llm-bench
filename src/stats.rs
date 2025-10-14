use chrono::{Timelike, Utc};
use metriken::histogram::Histogram;
use metriken_exposition::SnapshotterBuilder;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::time::{Instant, interval_at, timeout};

use crate::config::Config;
use crate::metrics::{
    ERRORS_CONNECTION, ERRORS_HTTP_4XX, ERRORS_HTTP_5XX, ERRORS_OTHER, ERRORS_PARSE,
    INTER_TOKEN_LATENCY, REQUEST_LATENCY, REQUESTS_FAILED, REQUESTS_INFLIGHT, REQUESTS_SENT,
    REQUESTS_SUCCESS, REQUESTS_TIMEOUT, RUNNING, TOKENS_INPUT, TOKENS_OUTPUT, TTFT,
};

/// Print with timestamp prefix
macro_rules! output {
    () => {
        let now = chrono::Utc::now();
        println!("{}", now.to_rfc3339_opts(chrono::SecondsFormat::Millis, false));
    };
    ($($arg:tt)*) => {{
        let now = chrono::Utc::now();
        print!("{} ", now.to_rfc3339_opts(chrono::SecondsFormat::Millis, false));
        println!($($arg)*);
    }};
}

struct MetricsSnapshot {
    // Store previous counter values
    requests_sent: u64,
    requests_success: u64,
    requests_failed: u64,
    requests_timeout: u64,
    tokens_input: u64,
    tokens_output: u64,
    errors_connection: u64,
    errors_4xx: u64,
    errors_5xx: u64,
    errors_parse: u64,
    errors_other: u64,

    // Store previous histogram snapshots
    ttft_histogram: Option<Histogram>,
    itl_histogram: Option<Histogram>,
    request_histogram: Option<Histogram>,
}

impl MetricsSnapshot {
    fn new() -> Self {
        Self {
            requests_sent: REQUESTS_SENT.value(),
            requests_success: REQUESTS_SUCCESS.value(),
            requests_failed: REQUESTS_FAILED.value(),
            requests_timeout: REQUESTS_TIMEOUT.value(),
            tokens_input: TOKENS_INPUT.value(),
            tokens_output: TOKENS_OUTPUT.value(),
            errors_connection: ERRORS_CONNECTION.value(),
            errors_4xx: ERRORS_HTTP_4XX.value(),
            errors_5xx: ERRORS_HTTP_5XX.value(),
            errors_parse: ERRORS_PARSE.value(),
            errors_other: ERRORS_OTHER.value(),
            ttft_histogram: TTFT.load(),
            itl_histogram: INTER_TOKEN_LATENCY.load(),
            request_histogram: REQUEST_LATENCY.load(),
        }
    }

    fn update(&mut self) {
        self.requests_sent = REQUESTS_SENT.value();
        self.requests_success = REQUESTS_SUCCESS.value();
        self.requests_failed = REQUESTS_FAILED.value();
        self.requests_timeout = REQUESTS_TIMEOUT.value();
        self.tokens_input = TOKENS_INPUT.value();
        self.tokens_output = TOKENS_OUTPUT.value();
        self.errors_connection = ERRORS_CONNECTION.value();
        self.errors_4xx = ERRORS_HTTP_4XX.value();
        self.errors_5xx = ERRORS_HTTP_5XX.value();
        self.errors_parse = ERRORS_PARSE.value();
        self.errors_other = ERRORS_OTHER.value();
        self.ttft_histogram = TTFT.load();
        self.itl_histogram = INTER_TOKEN_LATENCY.load();
        self.request_histogram = REQUEST_LATENCY.load();
    }
}

pub async fn periodic_stats(config: Config) {
    // Default to 1 minute interval if not specified
    let interval_duration = if let Some(metrics_config) = config.metrics.as_ref() {
        humantime::parse_duration(&metrics_config.interval).unwrap_or(Duration::from_secs(60))
    } else {
        Duration::from_secs(60)
    };

    // Build snapshotter for reading metrics (not used but kept for compatibility)
    let snapshotter = SnapshotterBuilder::new().build();

    // Get an aligned start time (aligned to the second)
    let start = Instant::now() - Duration::from_nanos(Utc::now().nanosecond() as u64)
        + Duration::from_secs(1);

    // Create interval timer
    let mut interval = interval_at(start, interval_duration);
    let mut window_id = 0;

    // Wait a bit for initial metrics
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Initialize previous snapshot
    let mut previous_snapshot = MetricsSnapshot::new();

    while RUNNING.load(Ordering::Relaxed) {
        // Use timeout to check RUNNING flag periodically
        if timeout(Duration::from_secs(1), interval.tick())
            .await
            .is_err()
        {
            continue;
        }

        // Take a snapshot for reading current values
        let _snapshot = snapshotter.snapshot();

        // Get current values
        let current_requests_sent = REQUESTS_SENT.value();
        let current_requests_success = REQUESTS_SUCCESS.value();
        let current_requests_failed = REQUESTS_FAILED.value();
        let current_requests_timeout = REQUESTS_TIMEOUT.value();
        let current_tokens_input = TOKENS_INPUT.value();
        let current_tokens_output = TOKENS_OUTPUT.value();
        let current_errors_connection = ERRORS_CONNECTION.value();
        let current_errors_4xx = ERRORS_HTTP_4XX.value();
        let current_errors_5xx = ERRORS_HTTP_5XX.value();
        let current_errors_parse = ERRORS_PARSE.value();
        let current_errors_other = ERRORS_OTHER.value();

        // Calculate deltas for this window
        let window_requests_sent = current_requests_sent - previous_snapshot.requests_sent;
        let window_requests_success = current_requests_success - previous_snapshot.requests_success;
        let window_requests_failed = current_requests_failed - previous_snapshot.requests_failed;
        let window_requests_timeout = current_requests_timeout - previous_snapshot.requests_timeout;
        let window_tokens_input = current_tokens_input - previous_snapshot.tokens_input;
        let window_tokens_output = current_tokens_output - previous_snapshot.tokens_output;
        let window_errors_connection =
            current_errors_connection - previous_snapshot.errors_connection;
        let window_errors_4xx = current_errors_4xx - previous_snapshot.errors_4xx;
        let window_errors_5xx = current_errors_5xx - previous_snapshot.errors_5xx;
        let window_errors_parse = current_errors_parse - previous_snapshot.errors_parse;
        let window_errors_other = current_errors_other - previous_snapshot.errors_other;

        // Skip window 0 since no requests have been sent yet
        if window_id == 0 {
            previous_snapshot.update();
            window_id += 1;
            continue;
        }

        // Print header with timestamp
        output!();
        output!("-----");
        output!("Window: {}", window_id);

        let requests_inflight = REQUESTS_INFLIGHT.value();
        let interval_secs = interval_duration.as_secs_f64();

        // Request statistics for this window (as rates)
        let sent_rate = window_requests_sent as f64 / interval_secs;
        output!(
            "Requests/s: Sent: {:.2} In-flight: {}",
            sent_rate,
            requests_inflight
        );

        // Response statistics for this window (as rates)
        let window_responses =
            window_requests_success + window_requests_failed + window_requests_timeout;
        let window_errors = window_requests_failed + window_requests_timeout;
        let responses_rate = window_responses as f64 / interval_secs;
        let success_rate_value = responses_rate - (window_errors as f64 / interval_secs);
        let error_rate = window_errors as f64 / interval_secs;
        let success_rate = if window_responses > 0 {
            100.0 * window_requests_success as f64 / window_responses as f64
        } else {
            0.0
        };
        output!(
            "Responses/s: Total: {:.2} Ok: {:.2} Err: {:.2} Success: {:.2}%",
            responses_rate,
            success_rate_value,
            error_rate,
            success_rate
        );

        // Error breakdown if any in this window (as rates)
        if window_requests_failed > 0 || window_requests_timeout > 0 {
            let conn_rate = window_errors_connection as f64 / interval_secs;
            let e4xx_rate = window_errors_4xx as f64 / interval_secs;
            let e5xx_rate = window_errors_5xx as f64 / interval_secs;
            let parse_rate = window_errors_parse as f64 / interval_secs;
            let timeout_rate = window_requests_timeout as f64 / interval_secs;
            let other_rate = window_errors_other as f64 / interval_secs;
            output!(
                "Errors/s: Connection: {:.2} 4xx: {:.2} 5xx: {:.2} Parse: {:.2} Timeout: {:.2} Other: {:.2}",
                conn_rate,
                e4xx_rate,
                e5xx_rate,
                parse_rate,
                timeout_rate,
                other_rate
            );
        }

        // Token statistics for this window (as rate per second)
        if window_tokens_input > 0 || window_tokens_output > 0 {
            let input_rate = window_tokens_input as f64 / interval_secs;
            let output_rate = window_tokens_output as f64 / interval_secs;
            output!(
                "Tokens/s: Input: {:.2} Output: {:.2}",
                input_rate,
                output_rate
            );
        }

        // Get current histograms
        let current_ttft = TTFT.load();
        let current_itl = INTER_TOKEN_LATENCY.load();
        let current_request = REQUEST_LATENCY.load();

        // TTFT percentiles for this window (using delta)
        if let (Some(current), Some(previous)) = (&current_ttft, &previous_snapshot.ttft_histogram)
        {
            if let Ok(delta) = current.wrapping_sub(previous)
                && let Ok(Some(percentiles)) = delta.percentiles(&[50.0, 90.0, 99.0])
                && percentiles.len() >= 3
            {
                let ttft_p50_ms = percentiles[0].1.end() / 1_000_000;
                let ttft_p90_ms = percentiles[1].1.end() / 1_000_000;
                let ttft_p99_ms = percentiles[2].1.end() / 1_000_000;

                output!(
                    "TTFT (ms): p50: {} p90: {} p99: {}",
                    ttft_p50_ms,
                    ttft_p90_ms,
                    ttft_p99_ms
                );
            }
        } else if let Some(current) = &current_ttft {
            // First window - use absolute values
            if let Ok(Some(percentiles)) = current.percentiles(&[50.0, 90.0, 99.0])
                && percentiles.len() >= 3
            {
                let ttft_p50_ms = percentiles[0].1.end() / 1_000_000;
                let ttft_p90_ms = percentiles[1].1.end() / 1_000_000;
                let ttft_p99_ms = percentiles[2].1.end() / 1_000_000;

                output!(
                    "TTFT (ms): p50: {} p90: {} p99: {}",
                    ttft_p50_ms,
                    ttft_p90_ms,
                    ttft_p99_ms
                );
            }
        }

        // ITL percentiles for this window (using delta)
        if let (Some(current), Some(previous)) = (&current_itl, &previous_snapshot.itl_histogram) {
            if let Ok(delta) = current.wrapping_sub(previous)
                && let Ok(Some(percentiles)) = delta.percentiles(&[50.0, 90.0, 99.0])
                && percentiles.len() >= 3
            {
                let itl_p50_ms = percentiles[0].1.end() / 1_000_000;
                let itl_p90_ms = percentiles[1].1.end() / 1_000_000;
                let itl_p99_ms = percentiles[2].1.end() / 1_000_000;

                if itl_p50_ms > 0 {
                    // Only show if we have ITL data
                    output!(
                        "ITL (ms): p50: {} p90: {} p99: {}",
                        itl_p50_ms,
                        itl_p90_ms,
                        itl_p99_ms
                    );
                }
            }
        } else if let Some(current) = &current_itl {
            // First window - use absolute values
            if let Ok(Some(percentiles)) = current.percentiles(&[50.0, 90.0, 99.0])
                && percentiles.len() >= 3
            {
                let itl_p50_ms = percentiles[0].1.end() / 1_000_000;
                let itl_p90_ms = percentiles[1].1.end() / 1_000_000;
                let itl_p99_ms = percentiles[2].1.end() / 1_000_000;

                if itl_p50_ms > 0 {
                    output!(
                        "ITL (ms): p50: {} p90: {} p99: {}",
                        itl_p50_ms,
                        itl_p90_ms,
                        itl_p99_ms
                    );
                }
            }
        }

        // Request latency percentiles for this window (using delta)
        if let (Some(current), Some(previous)) =
            (&current_request, &previous_snapshot.request_histogram)
        {
            if let Ok(delta) = current.wrapping_sub(previous)
                && let Ok(Some(percentiles)) = delta.percentiles(&[50.0, 90.0, 99.0])
                && percentiles.len() >= 3
            {
                let req_p50_ms = percentiles[0].1.end() / 1_000_000;
                let req_p90_ms = percentiles[1].1.end() / 1_000_000;
                let req_p99_ms = percentiles[2].1.end() / 1_000_000;

                output!(
                    "Request Latency (ms): p50: {} p90: {} p99: {}",
                    req_p50_ms,
                    req_p90_ms,
                    req_p99_ms
                );
            }
        } else if let Some(current) = &current_request {
            // First window - use absolute values
            if let Ok(Some(percentiles)) = current.percentiles(&[50.0, 90.0, 99.0])
                && percentiles.len() >= 3
            {
                let req_p50_ms = percentiles[0].1.end() / 1_000_000;
                let req_p90_ms = percentiles[1].1.end() / 1_000_000;
                let req_p99_ms = percentiles[2].1.end() / 1_000_000;

                output!(
                    "Request Latency (ms): p50: {} p90: {} p99: {}",
                    req_p50_ms,
                    req_p90_ms,
                    req_p99_ms
                );
            }
        }

        // Update previous snapshot
        previous_snapshot.update();

        window_id += 1;
    }
}

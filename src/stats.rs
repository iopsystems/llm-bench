use chrono::{Timelike, Utc};
use histogram::SampleQuantiles;
use metriken::histogram::Histogram;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::sync::Notify;
use tokio::time::{Instant, interval_at, timeout};

use crate::config::Config;
use crate::metrics::*;

/// Helper to read a CounterGroup value, defaulting to 0.
fn cg(group: &metriken::CounterGroup, idx: usize) -> u64 {
    group.value(idx).unwrap_or(0)
}

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
    requests_error: u64,
    requests_timeout: u64,
    requests_canceled: u64,
    tokens_input: u64,
    tokens_output: u64,
    errors_connection: u64,
    errors_4xx: u64,
    errors_5xx: u64,
    errors_parse: u64,
    errors_stream: u64,
    errors_other: u64,

    // Store previous histogram snapshots
    tpot_histogram: Option<Histogram>,
    request_histogram: Option<Histogram>,
}

impl MetricsSnapshot {
    fn merge_tpot() -> Option<Histogram> {
        let histograms = TPOT.load_all()?;
        let mut merged: Option<Histogram> = None;
        for h in histograms {
            merged = Some(match merged {
                Some(m) => m.wrapping_add(&h).ok()?,
                None => h,
            });
        }
        merged
    }

    fn new() -> Self {
        Self {
            requests_sent: cg(&REQUESTS, REQ_SENT),
            requests_success: cg(&REQUESTS, REQ_SUCCESS),
            requests_error: cg(&REQUESTS, REQ_ERROR),
            requests_timeout: cg(&REQUESTS, REQ_TIMEOUT),
            requests_canceled: cg(&REQUESTS, REQ_CANCELED),
            tokens_input: cg(&TOKENS, TOK_INPUT),
            tokens_output: cg(&TOKENS, TOK_OUTPUT_REASONING) + cg(&TOKENS, TOK_OUTPUT_CONTENT),
            errors_connection: cg(&ERRORS, ERR_CONNECTION),
            errors_4xx: cg(&ERRORS, ERR_HTTP_4XX),
            errors_5xx: cg(&ERRORS, ERR_HTTP_5XX),
            errors_parse: cg(&ERRORS, ERR_PARSE),
            errors_stream: cg(&ERRORS, ERR_STREAM),
            errors_other: cg(&ERRORS, ERR_OTHER),
            tpot_histogram: Self::merge_tpot(),
            request_histogram: REQUEST_LATENCY.load(),
        }
    }

    fn update(&mut self) {
        self.requests_sent = cg(&REQUESTS, REQ_SENT);
        self.requests_success = cg(&REQUESTS, REQ_SUCCESS);
        self.requests_error = cg(&REQUESTS, REQ_ERROR);
        self.requests_timeout = cg(&REQUESTS, REQ_TIMEOUT);
        self.requests_canceled = cg(&REQUESTS, REQ_CANCELED);
        self.tokens_input = cg(&TOKENS, TOK_INPUT);
        self.tokens_output = cg(&TOKENS, TOK_OUTPUT_REASONING) + cg(&TOKENS, TOK_OUTPUT_CONTENT);
        self.errors_connection = cg(&ERRORS, ERR_CONNECTION);
        self.errors_4xx = cg(&ERRORS, ERR_HTTP_4XX);
        self.errors_5xx = cg(&ERRORS, ERR_HTTP_5XX);
        self.errors_parse = cg(&ERRORS, ERR_PARSE);
        self.errors_stream = cg(&ERRORS, ERR_STREAM);
        self.errors_other = cg(&ERRORS, ERR_OTHER);
        self.tpot_histogram = Self::merge_tpot();
        self.request_histogram = REQUEST_LATENCY.load();
    }
}

pub async fn periodic_stats(config: Config, warmup_complete: Arc<Notify>) {
    // Default to 1 minute interval if not specified
    let interval_duration = if let Some(metrics_config) = config.metrics.as_ref() {
        humantime::parse_duration(&metrics_config.interval).unwrap_or(Duration::from_secs(60))
    } else {
        Duration::from_secs(60)
    };

    // Wait for warmup to complete before starting stats intervals
    warmup_complete.notified().await;

    // Get an aligned start time (aligned to the second) AFTER warmup
    let start = Instant::now() - Duration::from_nanos(Utc::now().nanosecond() as u64)
        + Duration::from_secs(1);

    // Create interval timer. Skip missed ticks rather than firing catch-up bursts
    // back-to-back, which would produce compressed windows whose rates (divided by
    // the actual elapsed time below) would otherwise be distorted.
    let mut interval = interval_at(start, interval_duration);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut window_id = 0;

    // Wait a bit for initial metrics
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Initialize previous snapshot
    let mut previous_snapshot = MetricsSnapshot::new();
    // Wall-clock time the current window's baseline was taken, so rates divide by
    // the ACTUAL elapsed time rather than the nominal interval (which drifts when
    // a tick is delayed under load).
    let mut last_window_time = Instant::now();

    while RUNNING.load(Ordering::Relaxed) {
        // Use timeout to check RUNNING flag periodically
        if timeout(Duration::from_secs(1), interval.tick())
            .await
            .is_err()
        {
            continue;
        }

        // Get current values
        let current_requests_sent = cg(&REQUESTS, REQ_SENT);
        let current_requests_success = cg(&REQUESTS, REQ_SUCCESS);
        let current_requests_error = cg(&REQUESTS, REQ_ERROR);
        let current_requests_timeout = cg(&REQUESTS, REQ_TIMEOUT);
        let current_requests_canceled = cg(&REQUESTS, REQ_CANCELED);
        let current_tokens_input = cg(&TOKENS, TOK_INPUT);
        let current_tokens_output =
            cg(&TOKENS, TOK_OUTPUT_REASONING) + cg(&TOKENS, TOK_OUTPUT_CONTENT);
        let current_errors_connection = cg(&ERRORS, ERR_CONNECTION);
        let current_errors_4xx = cg(&ERRORS, ERR_HTTP_4XX);
        let current_errors_5xx = cg(&ERRORS, ERR_HTTP_5XX);
        let current_errors_parse = cg(&ERRORS, ERR_PARSE);
        let current_errors_stream = cg(&ERRORS, ERR_STREAM);
        let current_errors_other = cg(&ERRORS, ERR_OTHER);

        // Calculate deltas for this window. saturating_sub because the counters
        // are read non-atomically across many statics — a concurrent increment
        // between reads could otherwise underflow.
        let window_requests_sent =
            current_requests_sent.saturating_sub(previous_snapshot.requests_sent);
        let window_requests_success =
            current_requests_success.saturating_sub(previous_snapshot.requests_success);
        let window_requests_error =
            current_requests_error.saturating_sub(previous_snapshot.requests_error);
        let window_requests_timeout =
            current_requests_timeout.saturating_sub(previous_snapshot.requests_timeout);
        let window_requests_canceled =
            current_requests_canceled.saturating_sub(previous_snapshot.requests_canceled);
        let window_tokens_input =
            current_tokens_input.saturating_sub(previous_snapshot.tokens_input);
        let window_tokens_output =
            current_tokens_output.saturating_sub(previous_snapshot.tokens_output);
        let window_errors_connection =
            current_errors_connection.saturating_sub(previous_snapshot.errors_connection);
        let window_errors_4xx = current_errors_4xx.saturating_sub(previous_snapshot.errors_4xx);
        let window_errors_5xx = current_errors_5xx.saturating_sub(previous_snapshot.errors_5xx);
        let window_errors_parse =
            current_errors_parse.saturating_sub(previous_snapshot.errors_parse);
        let window_errors_stream =
            current_errors_stream.saturating_sub(previous_snapshot.errors_stream);
        let window_errors_other =
            current_errors_other.saturating_sub(previous_snapshot.errors_other);

        // Skip window 0 since no requests have been sent yet
        if window_id == 0 {
            previous_snapshot.update();
            last_window_time = Instant::now();
            window_id += 1;
            continue;
        }

        // Print header with timestamp
        output!();
        output!("-----");
        output!("Window: {}", window_id);

        let requests_inflight = REQUESTS_INFLIGHT.value();
        // Divide by the ACTUAL elapsed time since the last window, not the nominal
        // interval, so a delayed tick doesn't distort the reported rates.
        let now = Instant::now();
        let interval_secs = (now - last_window_time).as_secs_f64().max(1e-9);
        last_window_time = now;

        // Request statistics for this window (as rates)
        let sent_rate = window_requests_sent as f64 / interval_secs;
        output!(
            "Requests/s: Sent: {:.2} In-flight: {}",
            sent_rate,
            requests_inflight
        );

        // Conversation progress (cumulative, not windowed)
        let conversations_sent = cg(&CONVERSATIONS, CONV_SENT);
        if conversations_sent > 0 {
            let conversations_success = cg(&CONVERSATIONS, CONV_SUCCESS);
            let turns = TURNS.value();
            output!(
                "Conversations: {} sent, {} complete, {} turns",
                conversations_sent,
                conversations_success,
                turns
            );
        }

        // Response statistics for this window (as rates)
        let window_responses = window_requests_success + window_requests_error;
        let responses_rate = window_responses as f64 / interval_secs;
        let success_rate_value = responses_rate - (window_requests_error as f64 / interval_secs);
        let error_rate = window_requests_error as f64 / interval_secs;
        let timeout_rate = window_requests_timeout as f64 / interval_secs;
        let success_rate = if window_responses > 0 {
            100.0 * window_requests_success as f64 / window_responses as f64
        } else {
            0.0
        };
        output!(
            "Responses/s: Total: {:.2} Ok: {:.2} Err: {:.2} Timeout: {:.2} Success: {:.2}%",
            responses_rate,
            success_rate_value,
            error_rate,
            timeout_rate,
            success_rate
        );
        if window_requests_canceled > 0 {
            let canceled_rate = window_requests_canceled as f64 / interval_secs;
            output!("Canceled/s: {:.2}", canceled_rate);
        }

        // Error breakdown if any in this window (as rates)
        if window_requests_error > 0 || window_requests_timeout > 0 {
            let conn_rate = window_errors_connection as f64 / interval_secs;
            let e4xx_rate = window_errors_4xx as f64 / interval_secs;
            let e5xx_rate = window_errors_5xx as f64 / interval_secs;
            let parse_rate = window_errors_parse as f64 / interval_secs;
            let timeout_rate = window_requests_timeout as f64 / interval_secs;
            let stream_rate = window_errors_stream as f64 / interval_secs;
            let other_rate = window_errors_other as f64 / interval_secs;
            output!(
                "Errors/s: Connection: {:.2} 4xx: {:.2} 5xx: {:.2} Parse: {:.2} Timeout: {:.2} Stream: {:.2} Other: {:.2}",
                conn_rate,
                e4xx_rate,
                e5xx_rate,
                parse_rate,
                timeout_rate,
                stream_rate,
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
        let current_tpot = MetricsSnapshot::merge_tpot();
        let current_request = REQUEST_LATENCY.load();

        // TPOT percentiles for this window (using delta)
        // wrapping_sub is safe because:
        // 1. Histograms only increment (additive operations)
        // 2. No individual bucket will double-wrap between samples
        //    - With 64-bit counters, this would require ~1.8e19 events
        //    - At 1M events/sec, this would take ~585 million years
        // 3. Each histogram bucket value in 'current' >= corresponding bucket in 'previous'
        if let (Some(current), Some(previous)) = (&current_tpot, &previous_snapshot.tpot_histogram)
        {
            if let Ok(delta) = current.wrapping_sub(previous)
                && let Ok(Some(result)) = delta.quantiles(&[0.5, 0.9, 0.95, 0.99])
            {
                let values: Vec<u64> = result
                    .entries()
                    .values()
                    .map(|b| b.end() / 1_000_000)
                    .collect();
                if values.len() == 4 && values[0] > 0 {
                    output!(
                        "TPOT (ms): p50: {} p90: {} p95: {} p99: {}",
                        values[0],
                        values[1],
                        values[2],
                        values[3]
                    );
                }
            }
        } else if let Some(current) = &current_tpot {
            // First window - use absolute values
            if let Ok(Some(result)) = current.quantiles(&[0.5, 0.9, 0.95, 0.99]) {
                let values: Vec<u64> = result
                    .entries()
                    .values()
                    .map(|b| b.end() / 1_000_000)
                    .collect();
                if values.len() == 4 && values[0] > 0 {
                    output!(
                        "TPOT (ms): p50: {} p90: {} p95: {} p99: {}",
                        values[0],
                        values[1],
                        values[2],
                        values[3]
                    );
                }
            }
        }

        // Request latency percentiles for this window (using delta)
        // wrapping_sub is safe because:
        // 1. Histograms only increment (additive operations)
        // 2. No individual bucket will double-wrap between samples
        //    - With 64-bit counters, this would require ~1.8e19 events
        //    - At 1M events/sec, this would take ~585 million years
        // 3. Each histogram bucket value in 'current' >= corresponding bucket in 'previous'
        if let (Some(current), Some(previous)) =
            (&current_request, &previous_snapshot.request_histogram)
        {
            if let Ok(delta) = current.wrapping_sub(previous)
                && let Ok(Some(result)) = delta.quantiles(&[0.5, 0.9, 0.95, 0.99])
            {
                let values: Vec<u64> = result
                    .entries()
                    .values()
                    .map(|b| b.end() / 1_000_000)
                    .collect();
                if values.len() == 4 {
                    output!(
                        "Request Latency (ms): p50: {} p90: {} p95: {} p99: {}",
                        values[0],
                        values[1],
                        values[2],
                        values[3]
                    );
                }
            }
        } else if let Some(current) = &current_request {
            // First window - use absolute values
            if let Ok(Some(result)) = current.quantiles(&[0.5, 0.9, 0.95, 0.99]) {
                let values: Vec<u64> = result
                    .entries()
                    .values()
                    .map(|b| b.end() / 1_000_000)
                    .collect();
                if values.len() == 4 {
                    output!(
                        "Request Latency (ms): p50: {} p90: {} p95: {} p99: {}",
                        values[0],
                        values[1],
                        values[2],
                        values[3]
                    );
                }
            }
        }

        // Update previous snapshot
        previous_snapshot.update();

        window_id += 1;
    }
}

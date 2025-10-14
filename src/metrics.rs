use metriken::{AtomicHistogram, Counter, Gauge, LazyCounter, LazyGauge, metric};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub enum RequestStatus {
    Success,
    Failed(ErrorType),
    Timeout,
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorType {
    Connection,
    Http4xx(u16),
    Http5xx(u16),
    Parse,
    Timeout,
    Other,
}

// Global running flag for background tasks
pub static RUNNING: AtomicBool = AtomicBool::new(false);

// Make metrics accessible for reporting
// In production, we'd use metriken-exposition properly

// Request metrics
#[metric(
    name = "requests",
    description = "Total number of requests",
    metadata = { status = "sent" }
)]
pub static REQUESTS_SENT: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "requests",
    description = "Successful requests",
    metadata = { status = "success" }
)]
pub static REQUESTS_SUCCESS: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "requests",
    description = "Failed requests",
    metadata = { status = "failed" }
)]
pub static REQUESTS_FAILED: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "requests",
    description = "Timed out requests",
    metadata = { status = "timeout" }
)]
pub static REQUESTS_TIMEOUT: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "requests",
    description = "Request retries",
    metadata = { status = "retried" }
)]
pub static REQUESTS_RETRIED: LazyCounter = LazyCounter::new(Counter::default);

// Error category metrics
#[metric(
    name = "errors",
    description = "Connection errors",
    metadata = { "type" = "connection" }
)]
pub static ERRORS_CONNECTION: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "errors",
    description = "HTTP 4xx errors",
    metadata = { "type" = "http_4xx" }
)]
pub static ERRORS_HTTP_4XX: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "errors",
    description = "HTTP 5xx errors",
    metadata = { "type" = "http_5xx" }
)]
pub static ERRORS_HTTP_5XX: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "errors",
    description = "Parse errors",
    metadata = { "type" = "parse" }
)]
pub static ERRORS_PARSE: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "errors",
    description = "Other errors",
    metadata = { "type" = "other" }
)]
pub static ERRORS_OTHER: LazyCounter = LazyCounter::new(Counter::default);

// Token metrics
#[metric(
    name = "tokens",
    description = "Input tokens processed",
    metadata = { direction = "input" }
)]
pub static TOKENS_INPUT: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "tokens",
    description = "Output tokens generated",
    metadata = { direction = "output" }
)]
pub static TOKENS_OUTPUT: LazyCounter = LazyCounter::new(Counter::default);

// Concurrency metrics
#[metric(
    name = "requests_inflight",
    description = "Current number of requests in flight"
)]
pub static REQUESTS_INFLIGHT: LazyGauge = LazyGauge::new(Gauge::default);

// Latency metrics (in nanoseconds)
// Histogram parameters: (grouping_power=5, max_value_power=64)
// This gives 32 buckets per power of 2, covering the full 64-bit range
#[metric(
    name = "ttft",
    description = "Time to first token in nanoseconds",
    metadata = { unit = "nanoseconds" }
)]
pub static TTFT: AtomicHistogram = AtomicHistogram::new(5, 64);

// Context-length-aware TTFT histograms
// Buckets based on production usage patterns
#[metric(
    name = "ttft_small",
    description = "TTFT for small contexts (0-200 tokens) - Simple Q&A",
    metadata = { unit = "nanoseconds", context_size = "small" }
)]
pub static TTFT_SMALL: AtomicHistogram = AtomicHistogram::new(5, 64);

#[metric(
    name = "ttft_medium",
    description = "TTFT for medium contexts (200-500 tokens) - Short conversations",
    metadata = { unit = "nanoseconds", context_size = "medium" }
)]
pub static TTFT_MEDIUM: AtomicHistogram = AtomicHistogram::new(5, 64);

#[metric(
    name = "ttft_large",
    description = "TTFT for large contexts (500-2000 tokens) - Technical/code help",
    metadata = { unit = "nanoseconds", context_size = "large" }
)]
pub static TTFT_LARGE: AtomicHistogram = AtomicHistogram::new(5, 64);

#[metric(
    name = "ttft_xlarge",
    description = "TTFT for extra large contexts (2000-8000 tokens) - Document analysis",
    metadata = { unit = "nanoseconds", context_size = "xlarge" }
)]
pub static TTFT_XLARGE: AtomicHistogram = AtomicHistogram::new(5, 64);

#[metric(
    name = "ttft_xxlarge",
    description = "TTFT for huge contexts (8000+ tokens) - Long context/RAG",
    metadata = { unit = "nanoseconds", context_size = "xxlarge" }
)]
pub static TTFT_XXLARGE: AtomicHistogram = AtomicHistogram::new(5, 64);

#[metric(
    name = "request_latency",
    description = "Total request latency in nanoseconds",
    metadata = { unit = "nanoseconds" }
)]
pub static REQUEST_LATENCY: AtomicHistogram = AtomicHistogram::new(5, 64);

// Inter-token latency
#[metric(
    name = "inter_token_latency",
    description = "Inter-token latency in nanoseconds",
    metadata = { unit = "nanoseconds" }
)]
pub static INTER_TOKEN_LATENCY: AtomicHistogram = AtomicHistogram::new(5, 64);

// Context-aware ITL histograms
#[metric(
    name = "itl_small",
    description = "Inter-token latency for small contexts (0-200 tokens)",
    metadata = { unit = "nanoseconds", context_size = "small" }
)]
pub static ITL_SMALL: AtomicHistogram = AtomicHistogram::new(5, 64);

#[metric(
    name = "itl_medium",
    description = "Inter-token latency for medium contexts (201-500 tokens)",
    metadata = { unit = "nanoseconds", context_size = "medium" }
)]
pub static ITL_MEDIUM: AtomicHistogram = AtomicHistogram::new(5, 64);

#[metric(
    name = "itl_large",
    description = "Inter-token latency for large contexts (501-1000 tokens)",
    metadata = { unit = "nanoseconds", context_size = "large" }
)]
pub static ITL_LARGE: AtomicHistogram = AtomicHistogram::new(5, 64);

#[metric(
    name = "itl_xlarge",
    description = "Inter-token latency for xlarge contexts (1001-2000 tokens)",
    metadata = { unit = "nanoseconds", context_size = "xlarge" }
)]
pub static ITL_XLARGE: AtomicHistogram = AtomicHistogram::new(5, 64);

#[metric(
    name = "itl_xxlarge",
    description = "Inter-token latency for xxlarge contexts (2001+ tokens)",
    metadata = { unit = "nanoseconds", context_size = "xxlarge" }
)]
pub static ITL_XXLARGE: AtomicHistogram = AtomicHistogram::new(5, 64);

pub struct Metrics;

impl Metrics {
    pub fn init() {
        // Metriken metrics are automatically registered via the #[metric] attribute
        // No explicit initialization needed
    }

    pub fn record_request_sent() {
        REQUESTS_SENT.increment();
        REQUESTS_INFLIGHT.increment();
    }

    pub fn record_request_complete(status: RequestStatus) {
        REQUESTS_INFLIGHT.decrement();
        match status {
            RequestStatus::Success => {
                REQUESTS_SUCCESS.increment();
            }
            RequestStatus::Failed(error_type) => {
                REQUESTS_FAILED.increment();
                match error_type {
                    ErrorType::Connection => ERRORS_CONNECTION.increment(),
                    ErrorType::Http4xx(_) => ERRORS_HTTP_4XX.increment(),
                    ErrorType::Http5xx(_) => ERRORS_HTTP_5XX.increment(),
                    ErrorType::Parse => ERRORS_PARSE.increment(),
                    ErrorType::Timeout => REQUESTS_TIMEOUT.increment(),
                    ErrorType::Other => ERRORS_OTHER.increment(),
                };
            }
            RequestStatus::Timeout => {
                REQUESTS_TIMEOUT.increment();
                ERRORS_OTHER.increment();
            }
        }
    }

    pub fn record_tokens(input: u64, output: u64) {
        TOKENS_INPUT.add(input);
        TOKENS_OUTPUT.add(output);
    }

    pub fn record_ttft(duration: Duration) {
        let _ = TTFT.increment(duration.as_nanos() as u64);
    }

    pub fn record_ttft_with_context(duration: Duration, input_tokens: u64) {
        let nanos = duration.as_nanos() as u64;

        // Record in overall histogram
        let _ = TTFT.increment(nanos);

        // Record in context-specific histogram based on production patterns
        match input_tokens {
            0..=200 => {
                let _ = TTFT_SMALL.increment(nanos); // ~50% of production traffic
            }
            201..=500 => {
                let _ = TTFT_MEDIUM.increment(nanos); // ~30% of production traffic
            }
            501..=2000 => {
                let _ = TTFT_LARGE.increment(nanos); // ~15% of production traffic
            }
            2001..=8000 => {
                let _ = TTFT_XLARGE.increment(nanos); // ~4% of production traffic
            }
            _ => {
                let _ = TTFT_XXLARGE.increment(nanos); // ~1% of production traffic
            }
        }
    }

    pub fn record_inter_token_latency(duration: Duration) {
        let _ = INTER_TOKEN_LATENCY.increment(duration.as_nanos() as u64);
    }

    pub fn record_itl_with_context(duration: Duration, input_tokens: u64) {
        let nanos = duration.as_nanos() as u64;

        // Record in overall histogram
        let _ = INTER_TOKEN_LATENCY.increment(nanos);

        // Record in context-specific histogram
        match input_tokens {
            0..=200 => {
                let _ = ITL_SMALL.increment(nanos);
            }
            201..=500 => {
                let _ = ITL_MEDIUM.increment(nanos);
            }
            501..=1000 => {
                let _ = ITL_LARGE.increment(nanos);
            }
            1001..=2000 => {
                let _ = ITL_XLARGE.increment(nanos);
            }
            _ => {
                let _ = ITL_XXLARGE.increment(nanos);
            }
        }
    }

    pub fn record_latency(duration: Duration) {
        let _ = REQUEST_LATENCY.increment(duration.as_nanos() as u64);
    }

    pub fn record_retry() {
        REQUESTS_RETRIED.increment();
    }
}

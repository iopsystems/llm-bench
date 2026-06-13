//! Saturation search state management.
//!
//! Finds the maximum concurrency an LLM server can handle while maintaining
//! SLO compliance on TTFT, ITL, and/or TPOT latency percentiles.

use crate::config::{SaturationConfig, SloPercentiles};
use crate::metrics;

use histogram::SampleQuantiles;
use metriken::HistogramGroup;
use metriken::histogram::Histogram;
use serde::Serialize;
use std::time::Instant;
use tokio::sync::Semaphore;

use std::sync::Arc;

/// Collected percentiles for a single step (used internally to reduce arg count).
struct StepPercentiles {
    ttft_p50: f64,
    ttft_p99: f64,
    ttft_p999: f64,
    itl_p50: f64,
    itl_p99: f64,
    itl_p999: f64,
    tpot_p50: f64,
    tpot_p99: f64,
    tpot_p999: f64,
}

/// Generous safety cap on total measured windows — real searches use far fewer;
/// this only guards against a non-terminating loop.
const SATURATION_WINDOW_BUDGET: u32 = 500;

/// What the async driver should do after a window completes.
pub enum AdvanceOutcome {
    /// Search finished.
    Completed,
    /// Measure the next window at `target`; drain/settle first when `drain` is set.
    Measure { target: usize, drain: bool },
    /// Drop to `to` and drain (no measurement); then call `resume` for the next step.
    Drain { to: usize },
}

/// State machine for concurrency-based saturation search. Owns the pure
/// [`SearchPlanner`] and the per-window metric snapshotting; the async driver in
/// the benchmark module applies the concurrency changes and drains.
pub struct SaturationSearchState {
    config: SaturationConfig,
    semaphore: Arc<Semaphore>,
    sample_window: std::time::Duration,
    planner: SearchPlanner,

    current_concurrency: usize,
    // Label + drain flag for the window currently being measured.
    current_phase: String,
    current_drained: bool,
    step_start: Instant,

    // Histogram snapshots at step start (for delta computation)
    step_ttft_snapshot: Option<Histogram>,
    step_itl_snapshot: Option<Histogram>,
    step_tpot_snapshot: Option<Histogram>,

    // Counter snapshots at step start
    step_output_tokens: u64,
    step_requests_success: u64,

    results: Vec<SaturationStep>,
    completed: bool,
    final_max: Option<usize>,
    header_printed: bool,
}

/// Result of a single concurrency step.
#[derive(Debug, Clone, Serialize)]
pub struct SaturationStep {
    pub concurrency: usize,
    pub duration_secs: f64,
    pub requests_completed: u64,
    pub output_tokens_per_sec: f64,
    pub requests_per_sec: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p99_ms: f64,
    pub ttft_p999_ms: f64,
    pub itl_p50_ms: f64,
    pub itl_p99_ms: f64,
    pub itl_p999_ms: f64,
    pub tpot_p50_ms: f64,
    pub tpot_p99_ms: f64,
    pub tpot_p999_ms: f64,
    pub slo_passed: bool,
    pub fail_reason: String,
    /// Which search phase produced this rung: "climb", "bisect", or "confirm".
    pub phase: String,
    /// True if concurrency was drained/settled before this window was measured.
    pub drained: bool,
}

/// Final saturation search results.
#[derive(Debug, Clone, Serialize)]
pub struct SaturationResults {
    pub max_compliant_concurrency: Option<usize>,
    /// First genuinely-failing concurrency (saturation onset), if reached.
    pub knee_concurrency: Option<usize>,
    /// Concurrencies that failed once but passed after draining — transient/
    /// metastable load, not a true capacity limit.
    pub transient_recoveries: Vec<usize>,
    pub steps: Vec<SaturationStep>,
}

impl SaturationSearchState {
    pub fn new(config: SaturationConfig, semaphore: Arc<Semaphore>) -> Self {
        let sample_window =
            humantime::parse_duration(&config.sample_window).expect("validated in Config");
        let planner = SearchPlanner::new(
            config.max_concurrency,
            config.step_multiplier,
            config.min_throughput_ratio,
            config.confirm_windows,
            SATURATION_WINDOW_BUDGET,
        );

        Self {
            config,
            semaphore,
            sample_window,
            planner,
            current_concurrency: 0, // set in initialize()
            current_phase: "climb".to_string(),
            current_drained: false,
            step_start: Instant::now(),
            step_ttft_snapshot: None,
            step_itl_snapshot: None,
            step_tpot_snapshot: None,
            step_output_tokens: 0,
            step_requests_success: 0,
            results: Vec::new(),
            completed: false,
            final_max: None,
            header_printed: false,
        }
    }

    pub fn sample_window(&self) -> std::time::Duration {
        self.sample_window
    }

    /// Set the starting concurrency and snapshot the first window.
    /// Must be called after warmup completes and before the control loop begins.
    pub fn initialize(&mut self) {
        self.current_concurrency = self.config.start_concurrency;
        self.current_phase = "climb".to_string();
        self.current_drained = false;
        self.begin_window();
    }

    /// True once the current window has run for a full `sample_window`.
    pub fn window_elapsed(&self) -> bool {
        !self.completed && self.step_start.elapsed() >= self.sample_window
    }

    /// Reset per-window metric snapshots and restart the window clock. Call after
    /// any (optional) drain so the measurement reflects the new steady state.
    pub fn begin_window(&mut self) {
        self.step_start = Instant::now();
        self.step_ttft_snapshot = merge_histogram_group(&metrics::TTFT);
        self.step_itl_snapshot = merge_histogram_group(&metrics::ITL);
        self.step_tpot_snapshot = merge_histogram_group(&metrics::TPOT);
        self.step_output_tokens = output_tokens_total();
        self.step_requests_success = metrics::REQUESTS.value(metrics::REQ_SUCCESS).unwrap_or(0);
    }

    /// Measure the just-finished window, advance the planner, apply the next
    /// concurrency to the semaphore, and report what the driver should do next.
    pub fn advance(&mut self) -> AdvanceOutcome {
        if !self.header_printed {
            print_header();
            self.header_printed = true;
        }

        let outcome = self.record_window();
        let action = self.planner.on_step(outcome);
        self.apply_action(action)
    }

    /// Follow-up after a `Drain` completes (no window was measured).
    pub fn resume(&mut self) -> AdvanceOutcome {
        let action = self.planner.resume();
        self.apply_action(action)
    }

    /// Apply a planner action to the semaphore and translate it for the driver.
    fn apply_action(&mut self, action: Action) -> AdvanceOutcome {
        match action {
            Action::Done { max_compliant } => {
                self.final_max = max_compliant;
                self.completed = true;
                print_summary(&self.results());
                AdvanceOutcome::Completed
            }
            Action::Measure { target, drain } => {
                self.apply_concurrency(target);
                self.current_phase = self.planner.phase_label().to_string();
                self.current_drained = drain;
                AdvanceOutcome::Measure { target, drain }
            }
            Action::Drain { to } => {
                self.apply_concurrency(to);
                AdvanceOutcome::Drain { to }
            }
        }
    }

    /// Grow (add_permits) or shrink (forget_permits) the semaphore to `target`.
    fn apply_concurrency(&mut self, target: usize) {
        use std::cmp::Ordering;
        match target.cmp(&self.current_concurrency) {
            Ordering::Greater => self
                .semaphore
                .add_permits(target - self.current_concurrency),
            Ordering::Less => {
                self.semaphore
                    .forget_permits(self.current_concurrency - target);
            }
            Ordering::Equal => {}
        }
        self.current_concurrency = target;
    }

    /// Compute the just-finished window's metrics, print + record the step, and
    /// return the outcome the planner consumes. Latency SLO drives pass/fail here;
    /// the planner applies the throughput (marginal-gain) gate from the throughput.
    fn record_window(&mut self) -> StepOutcome {
        let elapsed_secs = self.step_start.elapsed().as_secs_f64().max(1e-9);

        let current_ttft = merge_histogram_group(&metrics::TTFT);
        let current_itl = merge_histogram_group(&metrics::ITL);
        let current_tpot = merge_histogram_group(&metrics::TPOT);
        let current_output_tokens = output_tokens_total();
        let current_requests_success = metrics::REQUESTS.value(metrics::REQ_SUCCESS).unwrap_or(0);

        let delta_output_tokens = current_output_tokens.saturating_sub(self.step_output_tokens);
        let delta_requests = current_requests_success.saturating_sub(self.step_requests_success);
        let output_tokens_per_sec = delta_output_tokens as f64 / elapsed_secs;
        let requests_per_sec = delta_requests as f64 / elapsed_secs;

        let delta_ttft = compute_delta(&current_ttft, &self.step_ttft_snapshot);
        let delta_itl = compute_delta(&current_itl, &self.step_itl_snapshot);
        let delta_tpot = compute_delta(&current_tpot, &self.step_tpot_snapshot);

        let (ttft_p50, ttft_p99, ttft_p999) = extract_percentiles_ms(&delta_ttft);
        let (itl_p50, itl_p99, itl_p999) = extract_percentiles_ms(&delta_itl);
        let (tpot_p50, tpot_p99, tpot_p999) = extract_percentiles_ms(&delta_tpot);

        let percentiles = StepPercentiles {
            ttft_p50,
            ttft_p99,
            ttft_p999,
            itl_p50,
            itl_p99,
            itl_p999,
            tpot_p50,
            tpot_p99,
            tpot_p999,
        };

        let latency_reason = self.slo_fail_reason(&percentiles);
        let latency_ok = latency_reason.is_none();

        let step = SaturationStep {
            concurrency: self.current_concurrency,
            duration_secs: elapsed_secs,
            requests_completed: delta_requests,
            output_tokens_per_sec,
            requests_per_sec,
            ttft_p50_ms: ttft_p50,
            ttft_p99_ms: ttft_p99,
            ttft_p999_ms: ttft_p999,
            itl_p50_ms: itl_p50,
            itl_p99_ms: itl_p99,
            itl_p999_ms: itl_p999,
            tpot_p50_ms: tpot_p50,
            tpot_p99_ms: tpot_p99,
            tpot_p999_ms: tpot_p999,
            slo_passed: latency_ok,
            fail_reason: latency_reason.unwrap_or_default(),
            phase: self.current_phase.clone(),
            drained: self.current_drained,
        };

        print_step(self.results.len() + 1, &step);
        self.results.push(step);

        StepOutcome {
            concurrency: self.current_concurrency,
            latency_ok,
            throughput: output_tokens_per_sec,
        }
    }

    /// Check all configured SLO thresholds, returning the first violation found.
    fn slo_fail_reason(&self, percentiles: &StepPercentiles) -> Option<String> {
        if let Some(ref slo) = self.config.slo.ttft
            && let Some(reason) = check_percentile_slo(
                "TTFT",
                slo,
                percentiles.ttft_p50,
                percentiles.ttft_p99,
                percentiles.ttft_p999,
            )
        {
            return Some(reason);
        }
        if let Some(ref slo) = self.config.slo.itl
            && let Some(reason) = check_percentile_slo(
                "ITL",
                slo,
                percentiles.itl_p50,
                percentiles.itl_p99,
                percentiles.itl_p999,
            )
        {
            return Some(reason);
        }
        if let Some(ref slo) = self.config.slo.tpot
            && let Some(reason) = check_percentile_slo(
                "TPOT",
                slo,
                percentiles.tpot_p50,
                percentiles.tpot_p99,
                percentiles.tpot_p999,
            )
        {
            return Some(reason);
        }
        None
    }

    pub fn results(&self) -> SaturationResults {
        SaturationResults {
            max_compliant_concurrency: if self.completed { self.final_max } else { None },
            knee_concurrency: self.planner.knee_concurrency(),
            transient_recoveries: self.planner.transient_recoveries().to_vec(),
            steps: self.results.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_percentile_slo(
    metric_name: &str,
    slo: &SloPercentiles,
    p50: f64,
    p99: f64,
    p999: f64,
) -> Option<String> {
    if let Some(threshold) = slo.p50_ms
        && p50 > threshold
    {
        return Some(format!(
            "{} p50 {:.0}ms > {:.0}ms SLO",
            metric_name, p50, threshold
        ));
    }
    if let Some(threshold) = slo.p99_ms
        && p99 > threshold
    {
        return Some(format!(
            "{} p99 {:.0}ms > {:.0}ms SLO",
            metric_name, p99, threshold
        ));
    }
    if let Some(threshold) = slo.p999_ms
        && p999 > threshold
    {
        return Some(format!(
            "{} p999 {:.0}ms > {:.0}ms SLO",
            metric_name, p999, threshold
        ));
    }
    None
}

/// Merge all histograms in a HistogramGroup into a single Histogram.
fn merge_histogram_group(group: &HistogramGroup) -> Option<Histogram> {
    let histograms = group.load_all()?;
    let mut merged: Option<Histogram> = None;
    for h in histograms {
        merged = Some(match merged {
            Some(existing) => existing.checked_add(&h).unwrap_or(existing),
            None => h,
        });
    }
    merged
}

/// Compute a delta histogram (current - previous snapshot).
fn compute_delta(current: &Option<Histogram>, previous: &Option<Histogram>) -> Option<Histogram> {
    match (current, previous) {
        (Some(cur), Some(prev)) => cur.wrapping_sub(prev).ok(),
        (Some(cur), None) => Some(cur.clone()),
        _ => None,
    }
}

/// Extract (p50, p99, p999) from a histogram in milliseconds.
fn extract_percentiles_ms(histogram: &Option<Histogram>) -> (f64, f64, f64) {
    let Some(hist) = histogram else {
        return (0.0, 0.0, 0.0);
    };

    let mut p50 = 0.0;
    let mut p99 = 0.0;
    let mut p999 = 0.0;

    if let Ok(Some(result)) = hist.quantiles(&[0.5, 0.99, 0.999]) {
        let values: Vec<f64> = result
            .entries()
            .values()
            .map(|b| b.end() as f64 / 1_000_000.0)
            .collect();
        if values.len() == 3 {
            p50 = values[0];
            p99 = values[1];
            p999 = values[2];
        }
    }

    (p50, p99, p999)
}

/// Total output tokens (reasoning + content).
fn output_tokens_total() -> u64 {
    metrics::TOKENS
        .value(metrics::TOK_OUTPUT_REASONING)
        .unwrap_or(0)
        + metrics::TOKENS
            .value(metrics::TOK_OUTPUT_CONTENT)
            .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Console output
// ---------------------------------------------------------------------------

fn print_header() {
    println!();
    println!(
        "{:>6} | {:>12} | {:>10} | {:>10} | {:>12} | {:>12} | {:>12} | Result",
        "Step", "Concurrency", "Req/s", "Tokens/s", "TTFT p99", "ITL p99", "TPOT p99"
    );
    println!("{}", "-".repeat(101));
}

fn print_step(step_num: usize, step: &SaturationStep) {
    let verdict = if step.slo_passed {
        "PASS".to_string()
    } else {
        format!("FAIL ({})", step.fail_reason)
    };
    let drained = if step.drained { "*" } else { "" };
    let result = format!("[{}{}] {}", step.phase, drained, verdict);

    println!(
        "{:>6} | {:>12} | {:>10.2} | {:>10.1} | {:>10.0}ms | {:>10.0}ms | {:>10.0}ms | {}",
        step_num,
        step.concurrency,
        step.requests_per_sec,
        step.output_tokens_per_sec,
        step.ttft_p99_ms,
        step.itl_p99_ms,
        step.tpot_p99_ms,
        result,
    );
}

fn print_summary(results: &SaturationResults) {
    println!("{}", "-".repeat(101));
    println!();
    println!("Saturation Search Complete — Per-Step Summary");
    println!();
    print_header();
    for (i, step) in results.steps.iter().enumerate() {
        print_step(i + 1, step);
    }
    println!("{}", "-".repeat(101));
    println!();
    if let Some(max_c) = results.max_compliant_concurrency {
        let best_step = results
            .steps
            .iter()
            .filter(|s| s.slo_passed)
            .max_by(|a, b| {
                a.output_tokens_per_sec
                    .partial_cmp(&b.output_tokens_per_sec)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        println!("  Max compliant concurrency: {}", max_c);
        if let Some(step) = best_step {
            println!(
                "  Peak throughput: {:.1} tokens/s @ concurrency {}",
                step.output_tokens_per_sec, step.concurrency
            );
        }
    } else {
        println!("  No compliant concurrency found — SLO failed at start_concurrency");
    }
    if let Some(knee) = results.knee_concurrency {
        println!("  Saturation onset (knee): concurrency {}", knee);
    }
    if !results.transient_recoveries.is_empty() {
        println!(
            "  Transient recoveries (failed once, passed after drain): {:?}",
            results.transient_recoveries
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Search planner — pure decision logic for the concurrency search.
//
// Separated from all I/O (semaphore, metrics, printing) so the algorithm can be
// unit-tested without a live server. The impure driver measures one window per
// rung, builds a `StepOutcome`, calls `on_step`, and applies the returned `Action`.
// ---------------------------------------------------------------------------

/// Measured result of one concurrency rung.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOutcome {
    pub concurrency: usize,
    pub latency_ok: bool,
    pub throughput: f64,
}

/// Next thing the driver should do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    /// Set concurrency to `target` (grow/shrink the semaphore), draining+settling
    /// first when `drain` is true, then measure one window and call `on_step`.
    Measure { target: usize, drain: bool },
    /// Drop concurrency to `to` and drain to a clean slate WITHOUT measuring a
    /// window (used to clear a metastable/congested state before re-probing). The
    /// driver then calls `resume` for the follow-up action.
    Drain { to: usize },
    /// Search is complete.
    Done { max_compliant: Option<usize> },
}

#[derive(Debug, Clone)]
enum Phase {
    Climbing,
    /// Dropped to the last-good concurrency and drained; the next measurement
    /// re-probes `failed_at` from a clean slate to decide transient vs genuine.
    ReprobeMeasure {
        failed_at: usize,
    },
    /// Binary search; `low` passes, `high` fails, `base` is the fixed throughput
    /// baseline (the last-good rung when bisection began).
    Bisecting {
        low: usize,
        high: usize,
        base: (usize, f64),
    },
    /// Re-validate boundary `c` over several windows.
    Confirming {
        c: usize,
        passes: u32,
        total: u32,
    },
    Done,
}

/// Pure state machine driving the saturation search.
pub struct SearchPlanner {
    max_concurrency: usize,
    step_multiplier: f64,
    min_throughput_ratio: f64,
    confirm_windows: u32,
    window_budget: u32,
    windows_used: u32,
    phase: Phase,
    /// Highest known-good rung as (concurrency, throughput).
    last_good: Option<(usize, f64)>,
    reprobed: std::collections::HashSet<usize>,
    knee_concurrency: Option<usize>,
    transient_recoveries: Vec<usize>,
}

/// Throughput gate: the rung must deliver at least `min_ratio` of the throughput
/// linearly projected from `base` for its concurrency. At a plateau the actual
/// throughput stops scaling while the projection keeps rising, so the gate trips.
fn throughput_ok(min_ratio: f64, base: Option<(usize, f64)>, conc: usize, tput: f64) -> bool {
    match base {
        Some((bc, bt)) if bc > 0 && bt > 0.0 => {
            let expected = bt * (conc as f64 / bc as f64);
            tput >= min_ratio * expected
        }
        _ => true, // no usable baseline → can't judge throughput, let latency decide
    }
}

impl SearchPlanner {
    pub fn new(
        max_concurrency: usize,
        step_multiplier: f64,
        min_throughput_ratio: f64,
        confirm_windows: u32,
        window_budget: u32,
    ) -> Self {
        Self {
            max_concurrency,
            step_multiplier,
            min_throughput_ratio,
            confirm_windows: confirm_windows.max(1),
            window_budget,
            windows_used: 0,
            phase: Phase::Climbing,
            last_good: None,
            reprobed: std::collections::HashSet::new(),
            knee_concurrency: None,
            transient_recoveries: Vec::new(),
        }
    }

    pub fn knee_concurrency(&self) -> Option<usize> {
        self.knee_concurrency
    }

    pub fn transient_recoveries(&self) -> &[usize] {
        &self.transient_recoveries
    }

    /// Label for the phase that will produce the next measurement.
    pub fn phase_label(&self) -> &'static str {
        match self.phase {
            Phase::Climbing => "climb",
            Phase::ReprobeMeasure { .. } => "reprobe",
            Phase::Bisecting { .. } => "bisect",
            Phase::Confirming { .. } => "confirm",
            Phase::Done => "done",
        }
    }

    /// Follow-up action after a `Drain` completes (no window was measured). Only
    /// valid in the reprobe phase, where it re-measures `failed_at` from the now
    /// clean slate.
    pub fn resume(&self) -> Action {
        match self.phase {
            Phase::ReprobeMeasure { failed_at } => Action::Measure {
                target: failed_at,
                drain: true,
            },
            _ => Action::Done {
                max_compliant: self.last_good.map(|(c, _)| c),
            },
        }
    }

    /// Consume the just-measured rung's outcome and decide the next action.
    pub fn on_step(&mut self, o: StepOutcome) -> Action {
        self.windows_used += 1;
        if self.windows_used > self.window_budget {
            // Safety net against a non-terminating search; accept the best so far.
            self.phase = Phase::Done;
            return Action::Done {
                max_compliant: self.last_good.map(|(c, _)| c),
            };
        }

        match self.phase.clone() {
            Phase::Climbing => self.on_climb(o),
            Phase::ReprobeMeasure { failed_at } => self.on_reprobe(o, failed_at),
            Phase::Bisecting { low, high, base } => self.on_bisect(o, low, high, base),
            Phase::Confirming { c, passes, total } => self.on_confirm(o, c, passes, total),
            Phase::Done => Action::Done {
                max_compliant: self.last_good.map(|(c, _)| c),
            },
        }
    }

    fn on_climb(&mut self, o: StepOutcome) -> Action {
        let passed = o.latency_ok
            && throughput_ok(
                self.min_throughput_ratio,
                self.last_good,
                o.concurrency,
                o.throughput,
            );
        if passed {
            self.last_good = Some((o.concurrency, o.throughput));
            return self.climb_up(o.concurrency);
        }
        match self.last_good {
            None => {
                // Failed at the very first rung — nothing is compliant.
                self.knee_concurrency = Some(o.concurrency);
                self.phase = Phase::Done;
                Action::Done {
                    max_compliant: None,
                }
            }
            Some((g, _)) => {
                if self.reprobed.contains(&o.concurrency) {
                    self.knee_concurrency = Some(o.concurrency);
                    self.enter_bisect(g, o.concurrency)
                } else {
                    // Back off to the last-good rung and drain (no measurement)
                    // before re-probing, to distinguish a transient/metastable
                    // failure from a real one. The driver then `resume`s into the
                    // re-measurement of `failed_at`.
                    self.reprobed.insert(o.concurrency);
                    self.phase = Phase::ReprobeMeasure {
                        failed_at: o.concurrency,
                    };
                    Action::Drain { to: g }
                }
            }
        }
    }

    fn on_reprobe(&mut self, o: StepOutcome, failed_at: usize) -> Action {
        let passed = o.latency_ok
            && throughput_ok(
                self.min_throughput_ratio,
                self.last_good,
                o.concurrency,
                o.throughput,
            );
        if passed {
            // Recovered after draining → the first failure was transient.
            self.transient_recoveries.push(failed_at);
            self.last_good = Some((o.concurrency, o.throughput));
            self.climb_up(o.concurrency)
        } else {
            self.knee_concurrency = Some(failed_at);
            let g = self.last_good.map(|(c, _)| c).unwrap_or(0);
            self.enter_bisect(g, failed_at)
        }
    }

    fn climb_up(&mut self, c: usize) -> Action {
        let next = ((c as f64 * self.step_multiplier).ceil() as usize).max(c + 1);
        if next > self.max_concurrency {
            self.enter_confirm(c)
        } else {
            self.phase = Phase::Climbing;
            Action::Measure {
                target: next,
                drain: false,
            }
        }
    }

    fn enter_bisect(&mut self, low: usize, high: usize) -> Action {
        let base = self.last_good.unwrap_or((low, 0.0));
        self.bisect_advance(low, high, base)
    }

    fn bisect_advance(&mut self, low: usize, high: usize, base: (usize, f64)) -> Action {
        if high.saturating_sub(low) <= 1 {
            self.enter_confirm(low)
        } else {
            let mid = low + (high - low) / 2;
            self.phase = Phase::Bisecting { low, high, base };
            Action::Measure {
                target: mid,
                drain: true,
            }
        }
    }

    fn on_bisect(&mut self, o: StepOutcome, low: usize, high: usize, base: (usize, f64)) -> Action {
        let mid = o.concurrency;
        let passed =
            o.latency_ok && throughput_ok(self.min_throughput_ratio, Some(base), mid, o.throughput);
        let (nl, nh) = if passed {
            self.last_good = Some((mid, o.throughput));
            (mid, high)
        } else {
            (low, mid)
        };
        self.bisect_advance(nl, nh, base)
    }

    fn enter_confirm(&mut self, c: usize) -> Action {
        self.phase = Phase::Confirming {
            c,
            passes: 0,
            total: 0,
        };
        Action::Measure {
            target: c,
            drain: true,
        }
    }

    fn on_confirm(&mut self, o: StepOutcome, c: usize, passes: u32, total: u32) -> Action {
        let passes = passes + u32::from(o.latency_ok);
        let total = total + 1;
        if total >= self.confirm_windows {
            if passes * 2 > total {
                self.phase = Phase::Done;
                Action::Done {
                    max_compliant: Some(c),
                }
            } else if c <= 1 {
                self.phase = Phase::Done;
                Action::Done {
                    max_compliant: None,
                }
            } else {
                // Boundary unstable across windows — drop one and re-confirm.
                self.enter_confirm(c - 1)
            }
        } else {
            self.phase = Phase::Confirming { c, passes, total };
            Action::Measure {
                target: c,
                drain: false,
            }
        }
    }
}

#[cfg(test)]
mod planner_tests {
    use super::*;
    use std::collections::HashMap;

    /// Drive the planner against a model `(concurrency, nth_measurement) ->
    /// (latency_ok, throughput)` until it finishes, returning the result + planner.
    fn run<F: Fn(usize, usize) -> (bool, f64)>(
        start: usize,
        max: usize,
        mult: f64,
        min_ratio: f64,
        budget: u32,
        model: F,
    ) -> (Option<usize>, SearchPlanner) {
        let mut p = SearchPlanner::new(max, mult, min_ratio, 3, budget);
        let mut counts: HashMap<usize, usize> = HashMap::new();
        // Bootstrap: the driver measures `start` first, then feeds the planner.
        let mut next = Action::Measure {
            target: start,
            drain: false,
        };
        loop {
            match next {
                Action::Done { max_compliant } => return (max_compliant, p),
                // A Drain measures no window — just fetch the follow-up action.
                Action::Drain { .. } => next = p.resume(),
                Action::Measure { target, .. } => {
                    let n = counts.entry(target).or_insert(0);
                    *n += 1;
                    let (latency_ok, throughput) = model(target, *n);
                    next = p.on_step(StepOutcome {
                        concurrency: target,
                        latency_ok,
                        throughput,
                    });
                }
            }
        }
    }

    #[test]
    fn bisects_to_exact_latency_knee() {
        // Server passes for concurrency <= 50, fails above. Throughput linear.
        // A pure multiplicative climb (10,20,40,80) would report 40; bisection
        // must pin the exact ceiling of 50.
        let (mc, _) = run(10, 1000, 2.0, 0.9, 500, |c, _| (c <= 50, c as f64));
        assert_eq!(mc, Some(50));
    }

    #[test]
    fn no_compliant_when_start_fails() {
        let (mc, p) = run(10, 1000, 2.0, 0.9, 500, |c, _| (c <= 5, c as f64));
        assert_eq!(mc, None);
        assert_eq!(p.knee_concurrency(), Some(10));
    }

    #[test]
    fn transient_failure_recovers_after_drain() {
        // 80 fails the first time it's measured but is fine afterward; the real
        // latency knee is 200. The transient should be re-probed and recovered,
        // and the search should continue to the true knee.
        let (mc, p) = run(10, 1000, 2.0, 0.9, 500, |c, n| {
            let latency_ok = if c == 80 && n == 1 { false } else { c <= 200 };
            (latency_ok, c as f64)
        });
        assert_eq!(mc, Some(200));
        assert!(p.transient_recoveries().contains(&80));
    }

    #[test]
    fn detects_throughput_plateau() {
        // Latency always fine, but throughput plateaus at 100 tokens/s. Relative
        // to the pre-plateau baseline (80, 80), efficiency 100/c drops below 0.9
        // at c > 111, so the compliant ceiling is 111.
        let (mc, _) = run(10, 1000, 2.0, 0.9, 500, |c, _| (true, c.min(100) as f64));
        assert_eq!(mc, Some(111));
    }

    #[test]
    fn unstable_boundary_steps_down_on_confirm() {
        // 50 passes once (so bisection lands on it) but fails every later window,
        // so confirmation rejects it and steps down to a stable 49.
        let (mc, _) = run(10, 1000, 2.0, 0.9, 500, |c, n| {
            let ok = c <= 50 && !(c == 50 && n >= 2);
            (ok, c as f64)
        });
        assert_eq!(mc, Some(49));
    }
}

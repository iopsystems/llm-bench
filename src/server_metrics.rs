//! Scrapes a server's Prometheus /metrics on the snapshot interval and writes a
//! time-aligned server_metrics.parquet via llm-perf's existing MsgpackToParquet
//! pipeline. Converter ported from rezolus/src/recorder/prometheus.rs.

use metriken_exposition::{Counter, Gauge, Snapshot, SnapshotV2};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

#[derive(Clone, Hash, Eq, PartialEq)]
struct MetricKey {
    name: String,
    labels: Vec<(String, String)>,
}

/// Converts Prometheus text into metriken-exposition Snapshots, with stable
/// numeric metric IDs across scrapes (consistent parquet column identity) and
/// provenance metadata (metric name, labels, source, endpoint).
pub struct PrometheusConverter {
    metric_ids: HashMap<MetricKey, usize>,
    next_id: usize,
    source: String,
    endpoint: String,
}

impl PrometheusConverter {
    pub fn new(source: String, endpoint: String) -> Self {
        Self {
            metric_ids: HashMap::new(),
            next_id: 0,
            source,
            endpoint,
        }
    }

    fn get_or_assign_id(&mut self, name: &str, labels: &[(String, String)]) -> String {
        let key = MetricKey {
            name: name.to_string(),
            labels: labels.to_vec(),
        };
        if let Some(id) = self.metric_ids.get(&key) {
            return id.to_string();
        }
        let id = self.next_id;
        self.next_id += 1;
        self.metric_ids.insert(key, id);
        id.to_string()
    }

    fn build_metadata(&self, name: &str, labels: &[(String, String)]) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("metric".to_string(), name.to_string());
        for (k, v) in labels {
            m.insert(k.clone(), v.clone());
        }
        m.insert("source".to_string(), self.source.clone());
        m.insert("endpoint".to_string(), self.endpoint.clone());
        m
    }

    pub fn convert(&mut self, text: &str) -> Snapshot {
        let sanitized = sanitize_metric_names(text);
        let lines = sanitized.lines().map(|l| Ok(l.to_string()));
        let scrape = match prometheus_parse::Scrape::parse(lines) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("failed to parse prometheus metrics: {e}");
                return empty_snapshot();
            }
        };

        let mut counters = Vec::new();
        let mut gauges = Vec::new();

        for sample in scrape.samples {
            let mut labels: Vec<(String, String)> = sample
                .labels
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            labels.sort();

            match sample.value {
                prometheus_parse::Value::Counter(v) if v.is_finite() => {
                    let id = self.get_or_assign_id(&sample.metric, &labels);
                    counters.push(Counter {
                        name: id,
                        value: v as u64,
                        metadata: self.build_metadata(&sample.metric, &labels),
                    });
                }
                prometheus_parse::Value::Gauge(v) if v.is_finite() => {
                    let id = self.get_or_assign_id(&sample.metric, &labels);
                    gauges.push(Gauge {
                        name: id,
                        value: v as i64,
                        metadata: self.build_metadata(&sample.metric, &labels),
                    });
                }
                prometheus_parse::Value::Untyped(v) if v.is_finite() => {
                    let id = self.get_or_assign_id(&sample.metric, &labels);
                    let metadata = self.build_metadata(&sample.metric, &labels);
                    if sample.metric.ends_with("_total")
                        || sample.metric.ends_with("_sum")
                        || sample.metric.ends_with("_count")
                    {
                        counters.push(Counter {
                            name: id,
                            value: v as u64,
                            metadata,
                        });
                    } else {
                        gauges.push(Gauge {
                            name: id,
                            value: v as i64,
                            metadata,
                        });
                    }
                }
                prometheus_parse::Value::Summary(ref quantiles) => {
                    for q in quantiles {
                        if !q.count.is_finite() {
                            continue;
                        }
                        let mut ql = labels.clone();
                        ql.push(("quantile".to_string(), q.quantile.to_string()));
                        ql.sort();
                        let id = self.get_or_assign_id(&sample.metric, &ql);
                        gauges.push(Gauge {
                            name: id,
                            value: q.count as i64,
                            metadata: self.build_metadata(&sample.metric, &ql),
                        });
                    }
                }
                // Histograms are handled in a later task
                prometheus_parse::Value::Histogram(_) => {}
                _ => {}
            }
        }

        Snapshot::V2(SnapshotV2 {
            systemtime: SystemTime::now(),
            duration: Duration::ZERO,
            metadata: HashMap::new(),
            counters,
            gauges,
            histograms: Vec::new(),
        })
    }
}

/// Replace colons in metric names with underscores (prometheus-parse uses `\w+`
/// for names, excluding the colons namespaced exporters like vLLM use). Label
/// values and HELP text are left untouched.
fn sanitize_metric_names(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            if let Some(rest) = trimmed
                .strip_prefix("# HELP ")
                .or(trimmed.strip_prefix("# TYPE "))
            {
                let prefix = &trimmed[..trimmed.len() - rest.len()];
                let name_end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
                result.push_str(prefix);
                result.push_str(&rest[..name_end].replace(':', "_"));
                result.push_str(&rest[name_end..]);
            } else {
                result.push_str(trimmed);
            }
        } else {
            let name_end = trimmed
                .find(|c: char| c == '{' || c.is_whitespace())
                .unwrap_or(trimmed.len());
            result.push_str(&trimmed[..name_end].replace(':', "_"));
            result.push_str(&trimmed[name_end..]);
        }
        result.push('\n');
    }
    result
}

fn empty_snapshot() -> Snapshot {
    Snapshot::V2(SnapshotV2 {
        systemtime: SystemTime::now(),
        duration: Duration::ZERO,
        metadata: HashMap::new(),
        counters: Vec::new(),
        gauges: Vec::new(),
        histograms: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# HELP ferallm_decode_steps_total Decode steps executed.
# TYPE ferallm_decode_steps_total counter
ferallm_decode_steps_total 384
# TYPE ferallm_active_sequences gauge
ferallm_active_sequences 16
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running{model=\"llama\"} 12
";

    fn snap_parts(s: &Snapshot) -> (&Vec<Counter>, &Vec<Gauge>) {
        match s {
            Snapshot::V2(v2) => (&v2.counters, &v2.gauges),
            _ => panic!("expected V2 snapshot"),
        }
    }

    #[test]
    fn converts_counters_gauges_and_labels_with_provenance() {
        let mut c = PrometheusConverter::new("ferallm".into(), "http://x/metrics".into());
        let s = c.convert(SAMPLE);
        let (counters, gauges) = snap_parts(&s);
        assert_eq!(counters.len(), 1);
        assert_eq!(counters[0].value, 384);
        assert_eq!(
            counters[0].metadata.get("metric").unwrap(),
            "ferallm_decode_steps_total"
        );
        assert_eq!(counters[0].metadata.get("source").unwrap(), "ferallm");
        assert_eq!(gauges.len(), 2);
        let labeled = gauges
            .iter()
            .find(|g| g.metadata.get("metric").unwrap() == "vllm_num_requests_running")
            .unwrap();
        assert_eq!(labeled.value, 12);
        assert_eq!(labeled.metadata.get("model").unwrap(), "llama");
    }

    #[test]
    fn metric_ids_are_stable_across_scrapes() {
        let mut c = PrometheusConverter::new("s".into(), "e".into());
        let id1 = {
            let s = c.convert(SAMPLE);
            snap_parts(&s).0[0].name.clone()
        };
        let id2 = {
            let s = c.convert(SAMPLE);
            snap_parts(&s).0[0].name.clone()
        };
        assert_eq!(id1, id2);
    }
}

use llm_perf::config::Config;
use std::path::PathBuf;

#[test]
fn reproduce_apc_mitm_config() {
    let toml_content = r#"
[endpoint]
base_url = "http://127.0.0.1:8001/v1"
timeout = 300
max_tokens = 1

[input]
file = "synthetic"
sample_size = 10000

[input.synthetic]
prompt_tokens = 1000
common_prefix_sample_ratio = 1.0
common_prefix_tokens = 1000

[input.shared_prefix]
tokens = 1000
miss_rate = 0.0

[load]
arrival_distribution = "poisson"
concurrent_requests = 4
duration_seconds = 15
warmup_duration = 0

[metrics]
interval = "1s"
output = "llmperf.parquet"

[output]
file = "results.json"
format = "json"
"#;
    let path = PathBuf::from("/tmp/test_cfg_repro.toml");
    std::fs::write(&path, toml_content).unwrap();
    let cfg = Config::load(&path).unwrap();
    assert_eq!(cfg.input.shared_prefix.as_ref().unwrap().miss_rate, 0.0);
}

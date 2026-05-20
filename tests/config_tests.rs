use llm_perf::config::Config;
use std::path::PathBuf;

fn base_toml() -> &'static str {
    r#"
[endpoint]
base_url = "http://localhost:8080/v1"

[load]
total_requests = 10
concurrent_requests = 1

[input]
file = "examples/prompts/openorca-10000.jsonl"

[output]
format = "console"
"#
}

#[test]
fn system_prompt_inline_parses() {
    let toml = format!(
        "{}\n[input.system_prompt]\ncontent = \"You are helpful\"\n",
        base_toml()
    );
    let path = PathBuf::from("/tmp/sp_inline.toml");
    std::fs::write(&path, &toml).unwrap();
    let cfg = Config::load(&path).unwrap();
    let sp = cfg.input.system_prompt.unwrap();
    assert_eq!(sp.content.as_deref(), Some("You are helpful"));
    assert!(sp.file.is_none());
    assert!(sp.tokens.is_none());
}

#[test]
fn system_prompt_tokens_parses() {
    let toml = format!("{}\n[input.system_prompt]\ntokens = 512\n", base_toml());
    let path = PathBuf::from("/tmp/sp_tokens.toml");
    std::fs::write(&path, &toml).unwrap();
    let cfg = Config::load(&path).unwrap();
    let sp = cfg.input.system_prompt.unwrap();
    assert_eq!(sp.tokens, Some(512));
}

#[test]
fn system_prompt_multiple_sources_is_error() {
    let toml = format!(
        "{}\n[input.system_prompt]\ncontent = \"hi\"\ntokens = 64\n",
        base_toml()
    );
    let path = PathBuf::from("/tmp/sp_multi.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

#[test]
fn shared_prefix_parses() {
    let toml = format!(
        "{}\n[input.shared_prefix]\ntokens = 1024\nmiss_rate = 0.1\n",
        base_toml()
    );
    let path = PathBuf::from("/tmp/pfx.toml");
    std::fs::write(&path, &toml).unwrap();
    let cfg = Config::load(&path).unwrap();
    let pfx = cfg.input.shared_prefix.unwrap();
    assert_eq!(pfx.tokens, Some(1024));
    assert!((pfx.miss_rate - 0.1).abs() < 1e-9);
}

#[test]
fn shared_prefix_miss_rate_defaults_to_zero() {
    let toml = format!("{}\n[input.shared_prefix]\ntokens = 512\n", base_toml());
    let path = PathBuf::from("/tmp/pfx_default.toml");
    std::fs::write(&path, &toml).unwrap();
    let cfg = Config::load(&path).unwrap();
    assert_eq!(cfg.input.shared_prefix.unwrap().miss_rate, 0.0);
}

#[test]
fn shared_prefix_miss_rate_out_of_range_is_error() {
    let toml = format!(
        "{}\n[input.shared_prefix]\ntokens = 512\nmiss_rate = 1.5\n",
        base_toml()
    );
    let path = PathBuf::from("/tmp/pfx_bad.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

#[test]
fn unknown_field_in_input_is_error() {
    let toml = base_toml().replace("[input]\nfile", "[input]\nunknown_field = true\nfile");
    let path = PathBuf::from("/tmp/unknown.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

#[test]
fn unknown_field_at_top_level_is_error() {
    let toml = format!("{}\n[bogus_section]\nfoo = 1\n", base_toml());
    let path = PathBuf::from("/tmp/bogus_section.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

#[test]
fn cache_busting_field_is_error() {
    let toml = base_toml().replace("[input]\nfile", "[input]\ncache_busting = false\nfile");
    let path = PathBuf::from("/tmp/cb_field.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

#[test]
fn system_prompt_empty_table_is_error() {
    let toml = format!("{}\n[input.system_prompt]\n", base_toml());
    let path = PathBuf::from("/tmp/sp_empty.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

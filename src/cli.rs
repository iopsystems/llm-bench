use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "llm-bench")]
#[command(author, version, about = "Benchmark OpenAI-compatible LLM servers", long_about = None)]
pub struct Cli {
    /// Path to the TOML configuration file
    pub config: PathBuf,
}

impl Cli {
    pub fn parse_args() -> Self {
        Cli::parse()
    }
}

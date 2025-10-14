use anyhow::Result;
use llm_bench::{Cli, Config};
use log::{debug, info};
use ringlog::{File, LogBuilder, MultiLogBuilder, Output, Stderr};

/// Maximum log file size before rotation (10MB)
const LOG_FILE_MAX_SIZE: u64 = 1024 * 1024 * 10;

/// Log flush interval in milliseconds
const LOG_FLUSH_INTERVAL_MS: u64 = 100;

fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse_args();

    // Load configuration first to check verbosity setting
    let config = Config::load(&cli.config)?;

    // Set up logging with ringlog
    let log_level = config.log.level.to_level_filter();

    // Configure output destination
    let output: Box<dyn Output> = if let Some(ref log_file) = config.output.trace_log {
        // Log to file with rotation
        let backup_file = log_file.with_extension("old");
        Box::new(File::new(log_file.clone(), backup_file, LOG_FILE_MAX_SIZE)?)
    } else {
        // Log to stderr
        Box::new(Stderr::new())
    };

    // Build the base logger
    let base_log = LogBuilder::new()
        .output(output)
        .build()
        .expect("failed to initialize logger");

    // Create the multi-logger with level filtering
    let drain = MultiLogBuilder::new()
        .level_filter(log_level)
        .default(base_log)
        .build()
        .start();

    // Spawn thread to flush logs periodically
    std::thread::spawn(move || {
        let mut drain = drain;
        loop {
            std::thread::sleep(std::time::Duration::from_millis(LOG_FLUSH_INTERVAL_MS));
            let _ = drain.flush();
        }
    });

    // Print clean startup message
    if !config.output.quiet {
        println!("LLM Benchmark Tool");
        println!("   Config: {}", cli.config.display());
        println!("   Target: {}", config.endpoint.base_url);

        if let Some(qps) = config.load.qps {
            println!("   Mode: Fixed QPS ({:.1} req/s)", qps);
        } else {
            println!(
                "   Mode: Concurrent ({} workers)",
                config.load.concurrent_requests
            );
        }

        if let Some(total) = config.load.total_requests {
            println!("   Requests: {}", total);
        } else if let Some(duration) = config.load.duration_seconds {
            println!("   Duration: {}s", duration);
        }
        println!();
    }

    // Build custom tokio runtime with specified worker threads
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.runtime.worker_threads)
        .enable_all()
        .build()?;

    // Run the benchmark
    runtime.block_on(async { run_benchmark(config).await })
}

async fn run_benchmark(config: Config) -> Result<()> {
    // Start admin server if configured
    if let Some(ref admin_config) = config.admin
        && admin_config.enabled
    {
        let addr: std::net::SocketAddr = admin_config
            .listen
            .parse()
            .expect("Invalid admin listen address");

        info!("Starting metrics server on {}", addr);
        tokio::spawn(async move {
            llm_bench::admin::start_server(addr).await;
        });
    }

    debug!("Initializing benchmark runner");
    let runner = llm_bench::BenchmarkRunner::new(config).await?;
    info!("Starting benchmark run");
    runner.run().await?;
    info!("Benchmark completed successfully");
    Ok(())
}

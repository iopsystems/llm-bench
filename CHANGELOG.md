# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-10-14

### Bug Fixes

- Fixed success rate calculation in benchmark reports. Previously, success rate was calculated as `successful_requests / total_sent_requests`, which included in-flight requests that hadn't completed yet in duration-based tests. Now correctly calculated as `successful_requests / completed_requests`, providing an accurate success rate for completed requests only.

## [0.1.0] - 2025-01-15

### Initial Release

A high-performance benchmarking tool for OpenAI-compatible LLM inference servers. Designed to measure detailed performance characteristics of local LLM servers like llama-server, vLLM, TGI, and other OpenAI API-compatible endpoints.

#### Core Features

- **OpenAI API Compatibility**: Works with any server implementing `/v1/chat/completions` endpoint
- **Streaming Support**: Measures Time-To-First-Token (TTFT) via SSE streaming
- **Async/Concurrent Testing**: Configurable concurrent request handling with Tokio runtime
- **Token Counting**: Built-in tokenizer support for accurate token metrics
- **Automatic Retries**: Exponential backoff with jitter for transient failures

#### Load Patterns

- **Concurrent Mode**: Fixed number of concurrent workers
- **Fixed QPS Mode**: Maintain precise queries per second rate
- **Arrival Distributions**:
  - Uniform: Fixed intervals between requests (deterministic)
  - Poisson: Variable intervals following exponential distribution (realistic traffic)
- **Duration-Based Testing**: Run tests for specified time period
- **Request Count Mode**: Run fixed number of requests
- **Warmup Period**: Optional warmup phase to exclude cold start effects

#### Performance Metrics

- **Time to First Token (TTFT)**: Critical for streaming response UX
  - P50, P90, P95, P99 percentiles
  - Context-aware buckets (small, medium, large, xlarge, xxlarge)
- **Inter-Token Latency (ITL)**: Time between consecutive tokens
  - P50, P90, P95, P99 percentiles
  - Context-aware analysis by input size
- **End-to-End Request Latency**: Total request completion time
  - P50, P90, P95, P99 percentiles
- **Throughput Metrics**: Requests/s, input tokens/s, output tokens/s
- **Error Analysis**: Categorized tracking of connection, HTTP 4xx/5xx, timeout, and parse errors

#### Input Management

- **JSONL Input Format**: Simple prompt format with optional max_tokens
- **Prompt Cycling**: Automatically cycles through prompts for longer tests
- **Sample Size Control**: Limit prompts for quick tests
- **Shuffle Support**: Randomize prompt order for realistic patterns
- **Cache Busting**: Automatic per-request unique IDs to prevent response caching

#### Output Options

- **Console Output**: Clean, formatted results with detailed metrics
- **JSON Export**: Complete structured results with metadata for automation
- **Metrics Capture**: Periodic snapshots in Parquet format for time-series analysis

#### Observability

- **Efficient Logging**: Via `ringlog` with asynchronous ring buffer
- **Configurable Log Levels**: error, warn, info, debug, trace
- **Log File Support**: Automatic rotation at 10MB
- **Progress Indicators**: Real-time progress bars
- **Periodic Stats**: Optional runtime statistics display
- **Admin Metrics Server**: HTTP endpoint exposing metrics in Prometheus/JSON formats

#### Model Support

- **Auto-Detection**: Automatically detect model from server
- **Model Name Normalization**: Intelligent cleanup of paths and formats
- **Multi-Server Support**: Works with llama.cpp, vLLM, TGI, and other OpenAI-compatible servers

#### Configuration

- **TOML-Based**: Simple, readable configuration files
- **Comprehensive Examples**: Multiple scenario examples for different use cases
- **Real-World Datasets**: OpenOrca dataset with 10,000 diverse prompts included

#### Architecture

- **Rust 2024 Edition**: Modern Rust with latest language features
- **High Performance**: Efficient async I/O with Tokio
- **Connection Pooling**: Optimized HTTP client configuration
- **Production Ready**: Comprehensive error handling, proper resource cleanup
- **Well Documented**: Rustdoc comments on public API

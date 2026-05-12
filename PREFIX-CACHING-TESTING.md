# Testing vLLM Prefix Caching Effectiveness

This guide explains how to test the effectiveness of prefix caching in vLLM using the new synthetic prompt generation features.

## New Configuration Options

Two new fields have been added to `[input.synthetic]`:

- **`common_prefix_sample_ratio`** (default: `0.0`, range: `0.0` to `1.0`)
  - Controls the ratio of generated samples that share a common prefix
  - `0.0` = all prompts have unique prefixes (original behavior)
  - `1.0` = all prompts share the same common prefix
  - `0.5` = 50% of prompts share a common prefix

- **`common_prefix_tokens`** (default: `0`)
  - Controls the token length of the shared common prefix
  - Can be any value from `1` to `prompt_tokens`
  - If equals `prompt_tokens`, the entire prompt is the common prefix

## Quick Example

```toml
[input.synthetic]
prompt_tokens = 10000
add_prefix = false  # Disable unique prefixes
common_prefix_sample_ratio = 0.5  # 50% share common prefix
common_prefix_tokens = 5000  # 5000-token common prefix
```

This configuration generates prompts where:
- Each prompt is 10,000 tokens total
- 50% of prompts start with the same 5,000-token prefix
- The remaining 5,000 tokens are unique random text

## Testing Methodology

### Automated Testing Script

The `test-prefix-caching.sh` script automates the testing process:

```bash
./test-prefix-caching.sh
```

This script:
1. **Test 1**: Varies `common_prefix_sample_ratio` from 0.0 to 1.0 (in 0.1 steps) with a fixed 5000-token prefix
2. **Test 2**: Varies `common_prefix_tokens` from 0 to 10000 (in 1000-token steps) with a fixed 50% ratio

For each configuration, it:
- Runs a benchmark
- Fetches vLLM metrics from `http://localhost:8000/metrics`
- Calculates cache hit rate
- Saves results to CSV

Results are saved in `prefix-caching-results/results.csv`.

### Manual Testing

1. **Start your vLLM server** with prefix caching enabled:
   ```bash
   # Example vLLM startup command (adjust as needed)
   python -m vllm.entrypoints.openai.api_server \
       --model your-model \
       --enable-prefix-caching \
       --port 8000
   ```

2. **Create a test configuration** (see `test-prefix-cache-example.toml`):
   ```toml
   [endpoint]
   base_url = "http://localhost:8000/v1"
   max_tokens = 100

   [load]
   concurrent_requests = 10
   duration_seconds = 60

   [input]
   file = "synthetic"
   sample_size = 100

   [input.synthetic]
   prompt_tokens = 10000
   add_prefix = false
   common_prefix_sample_ratio = 0.5
   common_prefix_tokens = 5000

   [output]
   format = "console"
   ```

3. **Run the benchmark**:
   ```bash
   cargo run --release -- bench test-prefix-cache-example.toml
   ```

4. **Check vLLM metrics**:
   ```bash
   curl http://localhost:8000/metrics | grep -i cache
   ```

   Look for metrics like:
   - `vllm:cache_hit_total` or `vllm:prefix_cache_hit_total`
   - `vllm:cache_miss_total` or `vllm:prefix_cache_miss_total`
   - Cache hit rate = hits / (hits + misses)

### Verifying Generated Prompts

To inspect the generated prompts and verify the common prefix:

```bash
# Generate prompts to a JSONL file
cargo run --release -- generate-prompts test-prefix-cache-example.toml output.jsonl

# View first 100 characters of each prompt
jq -r '.prompt' output.jsonl | cut -c1-100
```

Prompts with the common prefix will start with identical text.

## Expected Results

### Cache Hit Rate vs Sample Ratio

With a fixed common prefix length (e.g., 5000 tokens):
- **ratio = 0.0**: ~0% cache hit rate (no shared prefixes)
- **ratio = 0.5**: ~25% cache hit rate (50% of requests hit the cache set up by the other 50%)
- **ratio = 1.0**: ~99% cache hit rate (all requests share the same prefix)

### Cache Hit Rate vs Prefix Length

With a fixed sample ratio (e.g., 50%):
- **prefix = 0**: ~0% cache hit rate (no prefix to cache)
- **prefix = small**: Low cache hit rate (small benefit)
- **prefix = large**: High cache hit rate (more tokens cached per hit)
- **prefix = prompt_tokens**: Maximum cache utilization

### Throughput Impact

Effective prefix caching should show:
- Increased throughput (tokens/sec)
- Reduced time to first token (TTFT)
- Reduced inter-token latency (ITL)

The improvement depends on:
- Cache hit rate
- Prefix length
- Model size
- Hardware capabilities

## Troubleshooting

### vLLM Metrics Not Available

If `curl http://localhost:8000/metrics` returns an error:
- Ensure vLLM is running and accessible
- Check if vLLM version supports metrics endpoint
- Try different metrics patterns (metric names vary by version)

### Cache Hit Rate is 0%

If you're not seeing cache hits:
- Verify prefix caching is enabled in vLLM
- Check that `common_prefix_sample_ratio > 0` and `common_prefix_tokens > 0`
- Ensure `add_prefix = false` (unique prefixes prevent caching)
- Increase warmup requests or duration to populate the cache

### Validation Errors

Common configuration errors:
- `common_prefix_tokens > prompt_tokens` (prefix can't exceed total)
- `common_prefix_sample_ratio > 0` but `common_prefix_tokens = 0` (need positive prefix length)
- Synthetic mode requires `endpoint.max_tokens` to be set

## Example Test Scenarios

### Scenario 1: Impact of Cache Coverage

Test how increasing the percentage of shared prefixes affects performance:

```bash
# 0% shared (baseline)
common_prefix_sample_ratio = 0.0

# 25% shared
common_prefix_sample_ratio = 0.25

# 50% shared
common_prefix_sample_ratio = 0.5

# 100% shared
common_prefix_sample_ratio = 1.0
```

### Scenario 2: Impact of Prefix Length

Test how prefix length affects caching benefit (with 50% coverage):

```bash
# Small prefix (10% of prompt)
common_prefix_tokens = 1000  # out of 10000

# Medium prefix (50% of prompt)
common_prefix_tokens = 5000

# Large prefix (90% of prompt)
common_prefix_tokens = 9000

# Full prompt
common_prefix_tokens = 10000
```

### Scenario 3: Real-World Simulation

Simulate a RAG (Retrieval-Augmented Generation) scenario:

```bash
# System prompt + context (7000 tokens) + unique query (3000 tokens)
prompt_tokens = 10000
common_prefix_sample_ratio = 1.0  # All requests have same system + context
common_prefix_tokens = 7000  # Only the query varies
```

## References

- vLLM Prefix Caching: https://docs.vllm.ai/en/latest/automatic_prefix_caching/index.html
- PR that added this feature: https://github.com/iopsystems/llm-perf/pull/79

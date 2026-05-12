#!/bin/bash
# Test script for vLLM prefix caching effectiveness
# This script runs benchmarks with varying common_prefix_sample_ratio and common_prefix_tokens
# and collects vLLM cache metrics after each run

set -e

# Configuration
VLLM_URL="http://localhost:8000"
VLLM_METRICS_URL="${VLLM_URL}/metrics"
PROMPT_TOKENS=10000
MAX_TOKENS=100
SAMPLE_SIZE=100
DURATION_SECONDS=60

# Output directory
OUTPUT_DIR="prefix-caching-results"
mkdir -p "$OUTPUT_DIR"

# Results file
RESULTS_FILE="${OUTPUT_DIR}/results.csv"
echo "ratio,prefix_tokens,cache_hits,cache_misses,hit_rate,throughput" > "$RESULTS_FILE"

# Function to extract cache metrics from vLLM
get_cache_metrics() {
    local metrics=$(curl -s "$VLLM_METRICS_URL")

    # Extract prefix cache metrics (tokens)
    # vLLM reports: prefix_cache_hits_total (cached tokens) and prefix_cache_queries_total (total queried tokens)
    local hits=$(echo "$metrics" | grep "vllm:prefix_cache_hits_total{" | grep -v "#" | awk '{print $2}')
    local queries=$(echo "$metrics" | grep "vllm:prefix_cache_queries_total{" | grep -v "#" | awk '{print $2}')

    # Default to 0 if not found
    hits=${hits:-0}
    queries=${queries:-0}

    # Calculate misses = queries - hits (use awk to handle scientific notation)
    local misses=$(awk -v q="$queries" -v h="$hits" 'BEGIN {printf "%.0f", q - h}')

    echo "$hits,$misses"
}

# Function to calculate hit rate
calc_hit_rate() {
    local hits=$1
    local misses=$2

    # Use awk for floating point arithmetic (handles scientific notation)
    awk -v h="$hits" -v m="$misses" 'BEGIN {
        total = h + m
        if (total == 0) {
            printf "0.00"
        } else {
            printf "%.2f", (h * 100) / total
        }
    }'
}

# Function to create and run a benchmark config
run_benchmark() {
    local ratio=$1
    local prefix_tokens=$2
    local config_file="${OUTPUT_DIR}/config-ratio-${ratio}-prefix-${prefix_tokens}.toml"

    echo "Running benchmark: ratio=$ratio, prefix_tokens=$prefix_tokens"

    # Create config file
    cat > "$config_file" <<EOF
[endpoint]
base_url = "${VLLM_URL}/v1"
timeout = 300
max_tokens = ${MAX_TOKENS}

[load]
concurrent_requests = 10
duration_seconds = ${DURATION_SECONDS}

[input]
file = "synthetic"
sample_size = ${SAMPLE_SIZE}

[input.synthetic]
prompt_tokens = ${PROMPT_TOKENS}
add_prefix = false
common_prefix_sample_ratio = ${ratio}
common_prefix_tokens = ${prefix_tokens}

[output]
format = "json"
file = "${OUTPUT_DIR}/results-ratio-${ratio}-prefix-${prefix_tokens}.json"
quiet = true
EOF

    # Get initial cache metrics (to calculate delta)
    local initial_metrics=$(get_cache_metrics)
    local initial_hits=$(echo "$initial_metrics" | cut -d',' -f1)
    local initial_misses=$(echo "$initial_metrics" | cut -d',' -f2)

    # Run benchmark
    cargo run --release -- bench "$config_file"

    # Get final cache metrics
    local final_metrics=$(get_cache_metrics)
    local final_hits=$(echo "$final_metrics" | cut -d',' -f1)
    local final_misses=$(echo "$final_metrics" | cut -d',' -f2)

    # Calculate deltas (use awk to handle scientific notation)
    local delta_hits=$(awk -v f="$final_hits" -v i="$initial_hits" 'BEGIN {printf "%.0f", f - i}')
    local delta_misses=$(awk -v f="$final_misses" -v i="$initial_misses" 'BEGIN {printf "%.0f", f - i}')
    local hit_rate=$(calc_hit_rate $delta_hits $delta_misses)

    # Extract throughput from results
    local throughput=$(cat "${OUTPUT_DIR}/results-ratio-${ratio}-prefix-${prefix_tokens}.json" | jq -r '.throughput.output_tokens_per_second' 2>/dev/null || echo "0")

    # Append to results file
    echo "${ratio},${prefix_tokens},${delta_hits},${delta_misses},${hit_rate},${throughput}" >> "$RESULTS_FILE"

    echo "  Cache hits: $delta_hits, misses: $delta_misses, hit rate: ${hit_rate}%"
    echo ""
}

echo "=== vLLM Prefix Caching Test ==="
echo "Testing with $PROMPT_TOKENS token prompts"
echo "Results will be saved to $OUTPUT_DIR"
echo ""

# Test 1: Vary sample ratio (0.0 to 1.0 in 0.1 steps) with fixed prefix tokens (5000)
echo "Test 1: Varying sample ratio (0.0 to 1.0) with 5000-token common prefix"
echo "-----------------------------------------------------------------------"
FIXED_PREFIX_TOKENS=5000
for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    run_benchmark "$ratio" "$FIXED_PREFIX_TOKENS"
    sleep 2  # Brief pause between tests
done

# Test 2: Vary prefix tokens (0 to 10000 in 1000 steps) with fixed ratio (0.5)
echo ""
echo "Test 2: Varying prefix token length (0 to 10000) with 50% sample ratio"
echo "------------------------------------------------------------------------"
FIXED_RATIO=0.5
for prefix_tokens in 0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
    # Skip 0 prefix tokens with non-zero ratio (invalid config)
    if [ $prefix_tokens -eq 0 ]; then
        echo "Skipping prefix_tokens=0 (invalid with ratio > 0)"
        continue
    fi
    run_benchmark "$FIXED_RATIO" "$prefix_tokens"
    sleep 2  # Brief pause between tests
done

echo ""
echo "=== Tests Complete ==="
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Summary:"
cat "$RESULTS_FILE" | column -t -s','

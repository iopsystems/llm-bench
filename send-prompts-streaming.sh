#!/bin/bash
# Send each prompt from identical-prompts.jsonl to vLLM using curl with STREAMING mode

ENDPOINT="http://localhost:8000/v1/chat/completions"
JSONL_FILE="identical-prompts.jsonl"

echo "=== Sending prompts to vLLM with STREAMING mode ==="
echo

request_num=0
while IFS= read -r line; do
    request_num=$((request_num + 1))
    
    # Extract prompt and max_tokens from JSONL
    prompt=$(echo "$line" | jq -r '.prompt')
    max_tokens=$(echo "$line" | jq -r '.max_tokens')
    
    echo "Request #$request_num (streaming):"
    echo "  Prompt length: ${#prompt} chars"
    echo "  Max tokens: $max_tokens"
    
    # Send streaming request
    response=$(curl -s -X POST "$ENDPOINT" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"Qwen/Qwen3-0.6B\",
            \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | jq -Rs .)}],
            \"max_tokens\": $max_tokens,
            \"stream\": true
        }")
    
    # Check if request was successful (streaming returns multiple SSE events)
    if echo "$response" | grep -q "data:"; then
        echo "  ✓ Success (streaming)"
    else
        echo "  ✗ Error: $response"
    fi
    
    echo
    sleep 0.1  # Small delay between requests
done < "$JSONL_FILE"

echo "=== All streaming requests completed ==="

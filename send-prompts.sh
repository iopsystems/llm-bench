#!/bin/bash
# Send each prompt from identical-prompts.jsonl to vLLM using curl

ENDPOINT="http://localhost:8000/v1/chat/completions"
JSONL_FILE="identical-prompts.jsonl"

echo "=== Sending prompts to vLLM one by one ==="
echo

request_num=0
while IFS= read -r line; do
    request_num=$((request_num + 1))
    
    # Extract prompt and max_tokens from JSONL
    prompt=$(echo "$line" | jq -r '.prompt')
    max_tokens=$(echo "$line" | jq -r '.max_tokens')
    
    echo "Request #$request_num:"
    echo "  Prompt length: ${#prompt} chars"
    echo "  Max tokens: $max_tokens"
    
    # Send request
    response=$(curl -s -X POST "$ENDPOINT" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"Qwen/Qwen3-0.6B\",
            \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | jq -Rs .)}],
            \"max_tokens\": $max_tokens,
            \"stream\": false
        }")
    
    # Check if request was successful
    if echo "$response" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
        echo "  ✓ Success"
    else
        echo "  ✗ Error: $response"
    fi
    
    echo
    sleep 0.1  # Small delay between requests
done < "$JSONL_FILE"

echo "=== All requests completed ==="

#!/usr/bin/env python3
"""
Download and prepare a real dataset for benchmarking.
This uses a subset of OpenOrca which has high-quality instruction-following examples.

Requirements:
    pip install pandas pyarrow requests
"""

import json
import random
import sys
from pathlib import Path

try:
    import pandas as pd
    import requests
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install pandas pyarrow requests")
    sys.exit(1)

def estimate_tokens(text):
    """Rough estimation: ~4 characters per token"""
    return len(text) // 4

def download_openorca_sample(num_samples=500):
    """Download and process OpenOrca dataset"""
    print("Downloading OpenOrca dataset from HuggingFace...")

    # Use the 1M-GPT4-Augmented subset (smaller, high quality)
    url = "https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/1M-GPT4-Augmented.parquet"

    # Download the parquet file
    print(f"Fetching {url}")
    print("(This may take a few minutes for the first download...)")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save to temporary file
        temp_file = Path("/tmp/openorca.parquet")
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded to {temp_file}")

        # Read the parquet file
        print("Reading parquet file...")
        df = pd.read_parquet(temp_file)

        print(f"Dataset contains {len(df)} examples")
        print(f"Columns: {list(df.columns)}")

        # Sample diverse prompts
        # OpenOrca has 'question' field for the prompt and 'response' for expected output
        sampled = df.sample(n=min(num_samples, len(df)), random_state=42)

        prompts = []
        for idx, row in sampled.iterrows():
            question = row.get('question', '')
            response = row.get('response', '')
            orca_id = row.get('id', f'orca-{idx}')

            if not question:
                continue

            # Estimate max_tokens based on response length
            # Add some buffer (1.5x) to allow for variation
            response_tokens = estimate_tokens(response)
            max_tokens = min(int(response_tokens * 1.5), 2500)

            # Ensure minimum of 50 tokens
            max_tokens = max(max_tokens, 50)

            prompts.append({
                "prompt": question,
                "max_tokens": max_tokens
            })

        print(f"\nProcessed {len(prompts)} prompts from OpenOrca")
        return prompts

    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        print("\nFalling back to synthetic examples...")
        return generate_synthetic_fallback(num_samples)
    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("\nFalling back to synthetic examples...")
        return generate_synthetic_fallback(num_samples)

def generate_synthetic_fallback(num_samples):
    """Generate synthetic examples if download fails"""
    print("Using synthetic fallback dataset")

    examples = [
        # Short Q&A
        {"prompt": "What is the difference between a violin and a fiddle?", "expected_tokens": 100},
        {"prompt": "How many planets are in our solar system?", "expected_tokens": 50},
        {"prompt": "What causes rain?", "expected_tokens": 150},
        {"prompt": "Explain TCP vs UDP in networking.", "expected_tokens": 200},
        {"prompt": "What is the Python GIL?", "expected_tokens": 250},

        # Code generation
        {"prompt": "Write a Python function to find the nth Fibonacci number using dynamic programming.", "expected_tokens": 200},
        {"prompt": "Create a SQL query to find the second highest salary from an employees table.", "expected_tokens": 150},
        {"prompt": "Implement a binary search tree in Python with insert and search methods.", "expected_tokens": 400},

        # Explanations
        {"prompt": "Explain how Bitcoin mining works to someone with no technical background.", "expected_tokens": 500},
        {"prompt": "What are the main differences between supervised and unsupervised learning? Provide examples.", "expected_tokens": 400},
        {"prompt": "Describe the process of photosynthesis in detail, including the light and dark reactions.", "expected_tokens": 600},

        # Creative tasks
        {"prompt": "Write a short story about a robot who discovers it can dream.", "expected_tokens": 800},
        {"prompt": "Analyze the pros and cons of remote work for both employees and employers.", "expected_tokens": 700},

        # Long-form
        {"prompt": "Create a comprehensive guide for beginners learning web development in 2024. Include roadmap, resources, and projects.", "expected_tokens": 2000},
        {"prompt": "Explain the history and impact of the Industrial Revolution, covering key inventions, social changes, and long-term effects.", "expected_tokens": 1500},
    ]

    prompts = []
    for i in range(num_samples):
        base = random.choice(examples)
        prompts.append({
            "prompt": base['prompt'],
            "max_tokens": min(base['expected_tokens'] + random.randint(-50, 100), 2500)
        })

    return prompts

def save_dataset(prompts, filename):
    """Save prompts in JSONL format"""
    with open(filename, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')
    print(f"Saved {len(prompts)} prompts to {filename}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download and prepare OpenOrca dataset for benchmarking')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate (default: 10000)')
    parser.add_argument('--output-dir', type=str, default='examples/prompts', help='Output directory (default: examples/prompts)')
    args = parser.parse_args()

    # Download and prepare dataset
    prompts = download_openorca_sample(args.samples)

    # Shuffle for variety
    random.shuffle(prompts)

    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset
    output_path = output_dir / f"openorca-{args.samples}.jsonl"
    save_dataset(prompts, str(output_path))

    print("\nDataset statistics:")
    token_counts = [p['max_tokens'] for p in prompts]
    print(f"   Total prompts: {len(prompts)}")
    print(f"   Average max_tokens: {sum(token_counts) / len(token_counts):.0f}")
    print(f"   Min max_tokens: {min(token_counts)}")
    print(f"   Max max_tokens: {max(token_counts)}")
    print(f"\nToken distribution:")
    print(f"   Prompts < 200 tokens: {sum(1 for t in token_counts if t < 200)} ({100*sum(1 for t in token_counts if t < 200)/len(token_counts):.1f}%)")
    print(f"   Prompts 200-500 tokens: {sum(1 for t in token_counts if 200 <= t < 500)} ({100*sum(1 for t in token_counts if 200 <= t < 500)/len(token_counts):.1f}%)")
    print(f"   Prompts 500-1000 tokens: {sum(1 for t in token_counts if 500 <= t < 1000)} ({100*sum(1 for t in token_counts if 500 <= t < 1000)/len(token_counts):.1f}%)")
    print(f"   Prompts > 1000 tokens: {sum(1 for t in token_counts if t >= 1000)} ({100*sum(1 for t in token_counts if t >= 1000)/len(token_counts):.1f}%)")

    print(f"\nDataset saved to: {output_path}")

if __name__ == "__main__":
    main()
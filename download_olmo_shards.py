#!/usr/bin/env python3
"""
Download OLMo pre-tokenized shards for training.

Usage:
    python download_olmo_shards.py --output-dir /path/to/data --num-params 1e9 --chinchilla-mult 4

This will download enough shards for a 1B model at 4x Chinchilla optimal.

The data will be saved preserving the directory structure expected by OLMo-core:
    output_dir/preprocessed/dolma3-0625/v0.1-official/{tokenizer}/...

When training, set mix_base_dir to your output_dir.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIX_FILE = os.path.join(SCRIPT_DIR, "src/olmo_core/data/mixes/OLMo-mix-0925-official.txt")
BASE_URL = "https://olmo-data.org"
TOKENIZER = "allenai/dolma3-tokenizer"
TOKENS_PER_PARAM = 20  # Chinchilla optimal
TOKENS_PER_SHARD = 145_000_000  # ~145M tokens per shard (580MB / 4 bytes)
BYTES_PER_SHARD = 580_000_000  # ~580MB per shard


def parse_num_params(s: str) -> int:
    """Parse parameter count from string like '1B', '1e9', '370M', etc."""
    s = s.strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1e9)
    elif s.endswith("M"):
        return int(float(s[:-1]) * 1e6)
    elif s.endswith("K"):
        return int(float(s[:-1]) * 1e3)
    else:
        return int(float(s))


def load_mix_paths(mix_file: str) -> list[tuple[str, str]]:
    """Load (label, path) pairs from mix file."""
    paths = []
    with open(mix_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            label, path = line.split(",")
            paths.append((label, path))
    return paths


def download_shard(url: str, output_path: Path, timeout: int = 600) -> tuple[bool, str]:
    """Download a single shard. Returns (success, message)."""
    try:
        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)

        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()

        # Write to temp file first, then rename
        tmp_path = output_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        tmp_path.rename(output_path)

        return True, f"Downloaded: {output_path.name}"
    except Exception as e:
        return False, f"Failed: {output_path.name} - {e}"


def main():
    parser = argparse.ArgumentParser(description="Download OLMo pre-tokenized shards")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base directory to save shards (will be your mix_base_dir)",
    )
    parser.add_argument(
        "--num-params",
        type=str,
        required=True,
        help="Number of parameters (e.g., '1B', '370M', '1e9')",
    )
    parser.add_argument(
        "--chinchilla-mult", type=float, default=4.0, help="Chinchilla multiple (default: 4.0)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of parallel downloads (default: 4)"
    )
    parser.add_argument("--mix-file", type=str, default=MIX_FILE, help="Path to mix file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be downloaded without downloading"
    )
    args = parser.parse_args()

    # Calculate requirements
    num_params = parse_num_params(args.num_params)
    tokens_needed = int(args.chinchilla_mult * TOKENS_PER_PARAM * num_params)
    shards_needed = (tokens_needed + TOKENS_PER_SHARD - 1) // TOKENS_PER_SHARD
    estimated_size_gb = shards_needed * BYTES_PER_SHARD / 1e9

    print(f"=== Download Configuration ===")
    print(f"Model size: {num_params:,} params ({num_params/1e9:.2f}B)")
    print(f"Chinchilla multiple: {args.chinchilla_mult}x")
    print(f"Tokens needed: {tokens_needed:,} ({tokens_needed/1e9:.1f}B)")
    print(f"Shards needed: {shards_needed:,}")
    print(f"Estimated size: {estimated_size_gb:.1f} GB")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load mix file
    all_paths = load_mix_paths(args.mix_file)
    if shards_needed > len(all_paths):
        print(f"WARNING: Requested {shards_needed} shards but mix only has {len(all_paths)}")
        shards_needed = len(all_paths)

    paths_to_use = all_paths[:shards_needed]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check which shards already exist
    # Preserve full directory structure for OLMo-core compatibility
    existing = []
    missing = []
    for label, path in paths_to_use:
        # Replace {TOKENIZER} placeholder and preserve full path
        resolved_path = path.replace("{TOKENIZER}", TOKENIZER)
        output_path = output_dir / resolved_path
        if output_path.exists():
            existing.append((label, path, output_path))
        else:
            missing.append((label, path, output_path))

    print(f"=== Shard Status ===")
    print(f"Already downloaded: {len(existing)}")
    print(f"Need to download: {len(missing)}")

    if len(existing) > 0:
        existing_size_gb = len(existing) * BYTES_PER_SHARD / 1e9
        print(f"Existing data: {existing_size_gb:.1f} GB")

    if len(missing) > 0:
        missing_size_gb = len(missing) * BYTES_PER_SHARD / 1e9
        print(f"Remaining to download: {missing_size_gb:.1f} GB")
    print()

    if len(missing) == 0:
        print("All shards already downloaded!")
        return

    if args.dry_run:
        print("=== Dry Run - Would download: ===")
        for label, path, output_path in missing[:10]:
            url = f"{BASE_URL}/{path.replace('{TOKENIZER}', TOKENIZER)}"
            print(f"  {output_path.relative_to(output_dir)}")
            print(f"    URL: {url}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print()
        print(f"Data will be saved to: {output_dir}")
        print(f"Use this as mix_base_dir when training.")
        return

    # Download missing shards
    print(f"=== Downloading {len(missing)} shards with {args.num_workers} workers ===")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for label, path, output_path in missing:
            url = f"{BASE_URL}/{path.replace('{TOKENIZER}', TOKENIZER)}"
            future = executor.submit(download_shard, url, output_path)
            futures[future] = output_path.name

        for future in as_completed(futures):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
            print(f"[{success_count + fail_count}/{len(missing)}] {msg}")

    print()
    print(f"=== Complete ===")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {fail_count}")

    total_shards = len(existing) + success_count
    total_tokens = total_shards * TOKENS_PER_SHARD
    print(f"Total shards available: {total_shards}")
    print(f"Total tokens available: {total_tokens:,} ({total_tokens/1e9:.1f}B)")
    print()
    print(f"To use this data, set mix_base_dir='{args.output_dir}'")


if __name__ == "__main__":
    main()

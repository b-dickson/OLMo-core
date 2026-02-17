#!/usr/bin/env python3
"""
Download OLMo PPL evaluation datasets from olmo-data.org

Usage:
    python download_eval_data.py --output-dir /data/user/dicksonb/data

The data will be saved to:
    output_dir/eval-data/perplexity/v3_small_dolma2/...

When training, set mix_base_dir to your output_dir.
"""

import argparse
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Constants
BASE_URL = "https://olmo-data.org"
TOKENIZER = "dolma2-tokenizer"  # For v3_small_ppl_validation with dolma2 tokenizer

# Eval datasets from v3-small-ppl-validation.txt
EVAL_DATASETS = [
    ("c4_en-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/c4_en/val/part-0-00000.npy"),
    ("dolma_books-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/dolma_books/val/part-0-00000.npy"),
    ("dolma_common-crawl-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/dolma_common-crawl/val/part-0-00000.npy"),
    ("dolma_pes2o-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/dolma_pes2o/val/part-0-00000.npy"),
    ("dolma_reddit-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/dolma_reddit/val/part-0-00000.npy"),
    ("dolma_stack-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/dolma_stack/val/part-0-00000.npy"),
    ("dolma_wiki-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/dolma_wiki/val/part-0-00000.npy"),
    ("ice-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/ice/val/part-0-00000.npy"),
    ("m2d2_s2orc-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/m2d2_s2orc/val/part-0-00000.npy"),
    ("pile-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/pile/val/part-0-00000.npy"),
    ("wikitext_103-validation", f"eval-data/perplexity/v3_small_{TOKENIZER}/wikitext_103/val/part-0-00000.npy"),
]


def download_file(url: str, output_path: Path, timeout: int = 300) -> tuple[bool, str]:
    """Download a single file. Returns (success, message)."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()

        tmp_path = output_path.with_suffix('.tmp')
        with open(tmp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        tmp_path.rename(output_path)

        return True, f"Downloaded: {output_path.name}"
    except Exception as e:
        return False, f"Failed: {output_path.name} - {e}"


def main():
    parser = argparse.ArgumentParser(description="Download OLMo PPL evaluation datasets")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Base directory to save data (will be your mix_base_dir)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel downloads (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Download PPL Eval Datasets ===")
    print(f"Source: {BASE_URL}")
    print(f"Output: {output_dir}")
    print(f"Datasets: {len(EVAL_DATASETS)}")
    print()

    # Check which files already exist
    existing = []
    missing = []
    for label, path in EVAL_DATASETS:
        output_path = output_dir / path
        if output_path.exists():
            existing.append((label, path, output_path))
        else:
            missing.append((label, path, output_path))

    print(f"Already downloaded: {len(existing)}")
    print(f"Need to download: {len(missing)}")
    print()

    if len(missing) == 0:
        print("All eval datasets already downloaded!")
        return

    if args.dry_run:
        print("=== Dry Run - Would download: ===")
        for label, path, output_path in missing:
            url = f"{BASE_URL}/{path}"
            print(f"  {label}")
            print(f"    -> {output_path.relative_to(output_dir)}")
        return

    # Download missing files
    print(f"=== Downloading {len(missing)} files ===")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for label, path, output_path in missing:
            url = f"{BASE_URL}/{path}"
            future = executor.submit(download_file, url, output_path)
            futures[future] = label

        for future in as_completed(futures):
            label = futures[future]
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
            print(f"[{success_count + fail_count}/{len(missing)}] {label}: {msg}")

    print()
    print(f"=== Complete ===")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {fail_count}")
    print()
    print(f"To use this data for eval, set mix_base_dir='{args.output_dir}'")


if __name__ == "__main__":
    main()

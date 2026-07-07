#!/usr/bin/env python3
"""Build (and optionally run) the Modal long-context command for a set of checkpoints.

Reads a checkpoint manifest (JSON list of {run, size, step, uri}) and emits a single
`modal run long_context_modal.py ...` command with clean per-checkpoint labels, so the
full suite launches with one call instead of pasting URIs by hand.

    # Print the command for all checkpoints in the manifest (review before launching):
    python launch_long_context_suite.py --manifest checkpoint_manifest.json --print

    # Actually launch (fans out one container per checkpoint):
    python launch_long_context_suite.py --manifest checkpoint_manifest.json \
        --gpu-type B200 --context-lengths 1024,2048,4096,8192,16384,32768 --run

    # Smoke subset (only 60M, tiny):
    python launch_long_context_suite.py --manifest checkpoint_manifest.json \
        --only-size 60M --num-samples 2 --context-lengths 1024,2048,4096 \
        --tasks niah_single --print
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ARCH_LABEL = {
    "vanilla_gated": "Full",
    "sliding_gated": "Sliding",
    "hybrid_gated_deltanet": "HGDN",
    "hc4": "HC4",
}


def label_for(run_name: str, size: str) -> str:
    arch = run_name.split("-")[0]
    # hybrid_gated_deltanet may appear with/without hc suffix; keep simple.
    for key, lab in ARCH_LABEL.items():
        if run_name.startswith(key):
            return f"{lab}-{size}"
    return f"{run_name}-{size}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="checkpoint_manifest.json")
    ap.add_argument("--only-size", default=None, help="Filter to one size, e.g. 60M.")
    ap.add_argument("--only-arch", default=None, help="Filter by run-name prefix, e.g. vanilla_gated.")
    ap.add_argument("--gpu-type", default="H100")
    ap.add_argument("--context-lengths", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--depths", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--tasks", default="niah_single,niah_multikey,niah_multivalue,variable_tracking")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output-dir", default="long_context_results")
    ap.add_argument("--modal-script", default=str(Path(__file__).with_name("long_context_modal.py")))
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--run", action="store_true", help="Actually invoke `modal run`.")
    args = ap.parse_args()

    entries = json.load(open(args.manifest))
    if args.only_size:
        entries = [e for e in entries if e["size"] == args.only_size]
    if args.only_arch:
        entries = [e for e in entries if e["run"].startswith(args.only_arch)]
    if not entries:
        sys.exit("No checkpoints match the filters.")

    uris = [e["uri"] for e in entries]
    labels = [label_for(e["run"], e["size"]) for e in entries]

    cmd = [
        "modal", "run", args.modal_script,
        "--checkpoint-uri", ",".join(uris),
        "--labels", ",".join(labels),
        "--gpu-type", args.gpu_type,
        "--context-lengths", args.context_lengths,
        "--depths", args.depths,
        "--tasks", args.tasks,
        "--num-samples", str(args.num_samples),
        "--batch-size", str(args.batch_size),
        "--dtype", args.dtype,
        "--output-dir", args.output_dir,
    ]

    print(f"# {len(entries)} checkpoint(s): " + ", ".join(labels), file=sys.stderr)
    if args.do_print or not args.run:
        print(" \\\n  ".join(shlex.quote(c) for c in cmd))
    if args.run:
        print("\n# launching...", file=sys.stderr)
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

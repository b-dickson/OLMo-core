#!/usr/bin/env python3
"""Modal launcher for WSDS and IsoFLOPs attention ladders.

This intentionally mirrors the ``lair`` launch style:
- read config rows from the existing ladder config files
- support array-style task selection
- support dry-run via script "dry-run" subcommand
- launch one task per Modal invocation
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WSDS_CONFIG = REPO_ROOT / "src/scripts/lair/attention_scaling_config.txt"
DEFAULT_ISOFLOPS_CONFIG = REPO_ROOT / "src/scripts/lair/isoflops_attention_scaling_config.txt"
REMOTE_REPO_PATH = "/repo/olmo-core"
REMOTE_LAUNCH_DIR = Path(REMOTE_REPO_PATH)
DEFAULT_SEQUENCE_LENGTH = 2048
MODAL_SECRET_NAME = "r2-creds"
app = modal.App("olmo-attn-ladders")


@dataclass
class WsdsConfigRow:
    attention_type: str
    size: str
    chinchilla_multiple: float
    sequence_length: int | None
    microbatch_discount: float | None


@dataclass
class IsoFlopsConfigRow:
    attention_type: str
    size: str
    target_flops: float
    sequence_length: int | None
    microbatch_discount: float | None


def _parse_task_indices(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(1, total + 1))
    indices: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start < 1 or end < start or end > total:
                raise ValueError(f"Invalid index range '{token}'")
            indices.extend(range(start, end + 1))
        else:
            idx = int(token)
            if idx < 1 or idx > total:
                raise ValueError(f"Invalid index '{token}'")
            indices.append(idx)
    return indices


def _parse_tail_numbers(values: list[str]) -> tuple[int | None, float | None]:
    seq_len: int | None = None
    microbatch: float | None = None
    if len(values) >= 1 and values[0]:
        if re.fullmatch(r"^\d+$", values[0]):
            seq_len = int(values[0])
        else:
            microbatch = float(values[0])
    if len(values) >= 2 and values[1]:
        microbatch = float(values[1]) if microbatch is None else microbatch
    return seq_len, microbatch


def _gpu_count_from_argv(default: int = 8) -> int:
    """Peek at sys.argv to determine GPU count for Modal resource allocation."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpus" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if arg.startswith("--gpus="):
            return int(arg.split("=", 1)[1])
    return default


_MODAL_GPU_COUNT = _gpu_count_from_argv()

_modal_image = (
    modal.Image.from_registry(
        os.environ.get("OLMO_DOCKER_IMAGE", "ghcr.io/allenai/olmo-core:latest"),
        add_python="3.12",
    )
    .add_local_dir(str(REPO_ROOT), remote_path=REMOTE_REPO_PATH)
)


@app.function(
    gpu=f"H100:{max(1, _MODAL_GPU_COUNT)}",
    image=_modal_image,
    timeout=172800,
    retries=0,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
)
def run_training_command(command: list[str], env_vars: dict[str, Any], run_name: str) -> None:
    merged_env = os.environ.copy()
    merged_env.update({k: str(v) for k, v in env_vars.items()})
    proc = subprocess.run(
        command,
        cwd=str(REMOTE_LAUNCH_DIR),
        env=merged_env,
        check=True,
        text=True,
    )
    print(f"Modal run finished for {run_name} (returncode={proc.returncode})")


def load_wsds_rows(config_file: Path) -> list[WsdsConfigRow]:
    rows: list[WsdsConfigRow] = []
    for line in config_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 3:
            raise ValueError(f"Malformed WSDS config line: {line}")
        attention_type, size, chinchilla_multiple = parts[:3]
        seq_len, microbatch = _parse_tail_numbers(parts[3:])
        rows.append(
            WsdsConfigRow(
                attention_type=attention_type,
                size=size,
                chinchilla_multiple=float(chinchilla_multiple),
                sequence_length=seq_len,
                microbatch_discount=microbatch,
            )
        )
    return rows


def load_isoflops_rows(config_file: Path) -> list[IsoFlopsConfigRow]:
    rows: list[IsoFlopsConfigRow] = []
    for line in config_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 3:
            raise ValueError(f"Malformed IsoFLOPs config line: {line}")
        attention_type, size, target_flops = parts[:3]
        seq_len, microbatch = _parse_tail_numbers(parts[3:])
        rows.append(
            IsoFlopsConfigRow(
                attention_type=attention_type,
                size=size,
                target_flops=float(target_flops),
                sequence_length=seq_len,
                microbatch_discount=microbatch,
            )
        )
    return rows


def _build_wsds_command(
    row: WsdsConfigRow,
    args: argparse.Namespace,
    task_id: int,
) -> tuple[list[str], str]:
    sequence_length = args.sequence_length or row.sequence_length or DEFAULT_SEQUENCE_LENGTH
    microbatch_discount = (
        args.microbatch_discount
        if args.microbatch_discount is not None
        else row.microbatch_discount
        if row.microbatch_discount is not None
        else 1.0
    )
    batch_size_multiplier = args.batch_size_multiplier or 1.0
    chinchilla_multiple = (
        args.chinchilla_multiple
        if args.chinchilla_multiple is not None
        else row.chinchilla_multiple
    )
    run_name = (
        f"{args.run_name_prefix}-{task_id}"
        if args.run_name_prefix
        else f"{row.attention_type}-{chinchilla_multiple}x-{row.size}_seq{sequence_length}_{task_id}"
    )

    command: list[str] = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={args.gpus}",
        "src/scripts/train/ladder/wsds_attention_ladder.py",
        "dry-run" if args.dry_run else "run",
        f"--name={run_name}",
        f"--size={row.size}",
        f"--attention-type={row.attention_type}",
        f"--chinchilla-multiple={chinchilla_multiple}",
        f"--batch-size-multiplier={batch_size_multiplier}",
        f"--microbatch-discount={microbatch_discount}",
        f"--sliding-window-size={args.sliding_window_size}",
        f"--train-data={args.train_data}",
        f"--eval-data-dir={args.eval_data_dir}",
        f"--sequence-length={sequence_length}",
        f"--max-gpus={args.gpus}",
        f"--wandb-entity={args.wandb_entity}",
        f"--project={args.project}",
    ]

    if args.optimizer:
        command.append(f"--optimizer={args.optimizer}")

    if args.dry_run and args.show_plot:
        command.append("--show-plot")

    return command, run_name


def _build_isoflops_command(
    row: IsoFlopsConfigRow,
    args: argparse.Namespace,
    task_id: int,
) -> tuple[list[str], str]:
    sequence_length = args.sequence_length or row.sequence_length or DEFAULT_SEQUENCE_LENGTH
    microbatch_discount = (
        args.microbatch_discount
        if args.microbatch_discount is not None
        else row.microbatch_discount
        if row.microbatch_discount is not None
        else 1.0
    )
    run_name = (
        f"{args.run_name_prefix}-{task_id}"
        if args.run_name_prefix
        else f"{row.attention_type}-{row.size}-{row.target_flops:g}_seq{sequence_length}_{task_id}"
    )

    command: list[str] = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={args.gpus}",
        "src/scripts/train/ladder/isoflops_attention_ladder.py",
        "dry-run" if args.dry_run else "run",
        f"--name={run_name}",
        f"--size={row.size}",
        f"--attention-type={row.attention_type}",
        f"--target-flops={row.target_flops}",
        f"--microbatch-discount={microbatch_discount}",
        f"--sequence-length={sequence_length}",
        f"--sliding-window-size={args.sliding_window_size}",
        f"--train-data={args.train_data}",
        f"--eval-data-dir={args.eval_data_dir}",
        f"--max-gpus={args.gpus}",
        f"--wandb-entity={args.wandb_entity}",
        f"--project={args.project}",
    ]

    if args.dry_run and args.show_plot:
        command.append("--show-plot")

    return command, run_name


def _collect_passthrough_env() -> dict[str, str]:
    keys = [
        "WANDB_MODE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
    ]
    return {k: v for k in keys if (v := os.getenv(k))}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch WSDS/IsoFLOPs ladder rows on Modal.")
    parser.add_argument(
        "--mode",
        choices=["wsds", "isoflops"],
        required=True,
        help="Which ladder to run.",
    )
    parser.add_argument(
        "--array",
        default=None,
        help="Array task indices, e.g. 1-12 or 1,3,7 (defaults to all).",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to ladder config file (defaults to attention_scaling_config.txt for wsds, isoflops_attention_scaling_config.txt for isoflops).",
    )
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs for torchrun.")
    parser.add_argument(
        "--train-data",
        required=False,
        default="r2://llm-data/nanochat/tokenized/*.npy",
        help="Training data glob.",
    )
    parser.add_argument(
        "--eval-data-dir",
        default="r2://llm-data/eval-data",
        help="Validation mix root directory.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default="iu-cogai",
        help="W&B entity.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Override sequence length from config file.",
    )
    parser.add_argument("--batch-size-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--chinchilla-multiple",
        type=float,
        default=None,
        help="Optional WSDS override for chinchilla multiple.",
    )
    parser.add_argument(
        "--microbatch-discount",
        type=float,
        default=None,
        help="Optional microbatch discount override.",
    )
    parser.add_argument(
        "--sliding-window-size",
        type=int,
        default=1024,
        help="Sliding window size argument for ladder scripts.",
    )
    parser.add_argument(
        "--optimizer",
        default="muon",
        help="WSDS optimizer override.",
    )
    parser.add_argument(
        "--ladder-root-dir",
        default=None,
        help="Optional checkpoint root override (maps to LADDER_ROOT_DIR).",
    )
    parser.add_argument(
        "--run-name-prefix",
        default=None,
        help="Optional explicit run-name prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Pass through dry-run instead of run.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        default=False,
        help="For dry-runs, display LR plots.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.project is None:
        args.project = (
            "attn-scaling-isoflops" if args.mode == "isoflops" else "attn-scaling-ladder"
        )
    env_vars = _collect_passthrough_env()
    if args.ladder_root_dir:
        env_vars["LADDER_ROOT_DIR"] = args.ladder_root_dir

    if args.config_file:
        config_path = Path(args.config_file)
    elif args.mode == "wsds":
        config_path = DEFAULT_WSDS_CONFIG
    else:
        config_path = DEFAULT_ISOFLOPS_CONFIG

    if args.mode == "wsds":
        rows = load_wsds_rows(config_path)
    else:
        rows = load_isoflops_rows(config_path)

    task_ids = _parse_task_indices(args.array, len(rows))
    if not task_ids:
        raise SystemExit("No tasks selected.")

    launch_fn = _build_wsds_command if args.mode == "wsds" else _build_isoflops_command
    handles = []

    if args.dry_run:
        print("Modal dry-run: no remote submission; commands below for inspection.")

    print(f"Modal launcher starting: mode={args.mode}, selected={len(task_ids)} tasks, gpus={args.gpus}")
    for idx in task_ids:
        row = rows[idx - 1]
        command, run_name = launch_fn(row, args, idx)

        command_str = " ".join(shlex.quote(c) for c in command)
        if args.dry_run:
            print(f"[{idx}] {run_name}")
            print(f"  {command_str}")
            continue

        handle = run_training_command.remote(command, env_vars, run_name)
        handles.append((idx, run_name, handle))
        print(f"Submitted task {idx} -> {run_name}")

    if args.dry_run:
        print("Dry-run complete. No tasks submitted.")
        return

    for idx, run_name, handle in handles:
        print(f"Waiting for task {idx} ({run_name})")
        try:
            handle.result()
            print(f"Done: task {idx} ({run_name})")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed: task {idx} ({run_name}) => {exc}")
            raise


@app.local_entrypoint()
def local_entrypoint(*args: str) -> None:
    main(list(args))


if __name__ == "__main__":
    main(sys.argv[1:])

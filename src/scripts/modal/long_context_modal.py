#!/usr/bin/env python3
"""Modal launcher for long-context (NIAH / RULER) evaluation of trained checkpoints.

Mirrors ``attention_ladder_modal.py``: same base image, R2 credentials secret, and
single-GPU container. Each invocation pulls one checkpoint (from R2/S3 or a local
mount), runs ``scripts.eval.long_context.run_long_context`` inside the container, and
returns / uploads the results JSON.

Example
-------
    # One checkpoint (note the r2:// scheme, resolved natively by olmo-core):
    modal run long_context_modal.py \
        --checkpoint-uri r2://llm-data/checkpoints/model-ladders/vanilla_gated-4.0x-370M_seq8192_4/370M/step26000 \
        --label "Full-370M" --context-lengths "1024,2048,4096,8192,16384"

    # Several checkpoints (comma-separated), fanned out as parallel containers:
    modal run long_context_modal.py \
        --checkpoint-uri "r2://llm-data/checkpoints/model-ladders/hybrid_gated_deltanet-4.0x-370M_seq8192_.../370M/stepN,r2://llm-data/checkpoints/model-ladders/sliding_gated-4.0x-370M_seq8192_.../370M/stepN" \
        --labels "HGDN-370M,Sliding-370M"

Results are written to ``--output-dir`` (R2/S3 URI) if given, else returned to the
local caller and saved under ``./long_context_results/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import modal

try:
    REPO_ROOT = Path(__file__).resolve().parents[3]
except IndexError:
    REPO_ROOT = Path("/repo/olmo-core")
REMOTE_REPO_PATH = "/repo/olmo-core"
REMOTE_LAUNCH_DIR = Path(REMOTE_REPO_PATH)
MODAL_SECRET_NAME = "r2-creds"
app = modal.App("olmo-long-context")


# ── R2 credential file writer (copied from attention_ladder_modal.py) ──────────
def _write_aws_profile_files(aws_dir: Path, r2_profile: str, r2_endpoint: str,
                             access_key: str, secret_key: str,
                             session_token: Optional[str] = None,
                             region: Optional[str] = None) -> None:
    aws_dir.mkdir(parents=True, exist_ok=True)
    config_file = aws_dir / "config"
    config_contents = [f"[profile {r2_profile}]", f"region = {region or 'auto'}"]
    if r2_endpoint:
        config_contents.append(f"endpoint_url = {r2_endpoint}")
    config_file.write_text("\n".join(config_contents) + "\n")
    if access_key and secret_key:
        creds = [f"[{r2_profile}]", f"aws_access_key_id = {access_key}",
                 f"aws_secret_access_key = {secret_key}"]
        if session_token:
            creds.append(f"aws_session_token = {session_token}")
        (aws_dir / "credentials").write_text("\n".join(creds) + "\n")


def _resolve_r2_profile(env: dict) -> str:
    return env.get("R2_PROFILE") or env.get("AWS_PROFILE") or "r2"


def _gpu_type_from_argv(default: str = "H100") -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu-type" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--gpu-type="):
            return arg.split("=", 1)[1]
    return default


_MODAL_GPU_TYPE = _gpu_type_from_argv()
_docker_image = os.environ.get("OLMO_DOCKER_IMAGE", "ghcr.io/allenai/olmo-core:latest")
_registry_secret = (
    modal.Secret.from_name("ghcr-creds") if "ghcr.io/b-dickson" in _docker_image else None
)
_modal_image = (
    modal.Image.from_registry(_docker_image, add_python="3.12", secret=_registry_secret)
    .run_commands("pip install uv")
    .add_local_dir(str(REPO_ROOT), remote_path=REMOTE_REPO_PATH)
)


@app.function(
    gpu=f"{_MODAL_GPU_TYPE}:1",
    image=_modal_image,
    timeout=43200,
    retries=0,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
)
def run_eval(checkpoint_uri: str, label: str, cli: dict[str, Any]) -> dict:
    """Container entrypoint: install package, set up R2, run the long-context eval."""
    import subprocess

    print(f"[Modal] Installing olmo-core for long-context eval ({label})...")
    sys.stdout.flush()
    subprocess.check_call(["uv", "pip", "install", "-e", ".[all]", "--system"],
                          cwd=str(REMOTE_LAUNCH_DIR))

    merged_env = os.environ.copy()

    # PyTorch bundled NVRTC before host driver (B200 CUDA 13 vs container CUDA 12).
    for _pyver in ("3.12", "3.11"):
        _nvrtc = f"/opt/conda/lib/python{_pyver}/site-packages/nvidia/cuda_nvrtc/lib"
        if os.path.isdir(_nvrtc):
            merged_env["LD_LIBRARY_PATH"] = _nvrtc + ":" + merged_env.get("LD_LIBRARY_PATH", "")
            break

    # Write R2 profile so from_checkpoint can read s3://... URIs.
    r2_profile = _resolve_r2_profile(merged_env)
    _write_aws_profile_files(
        aws_dir=Path.home() / ".aws",
        r2_profile=r2_profile,
        r2_endpoint=merged_env.get("R2_ENDPOINT_URL", ""),
        access_key=merged_env.get("AWS_ACCESS_KEY_ID") or merged_env.get("R2_ACCESS_KEY_ID", ""),
        secret_key=merged_env.get("AWS_SECRET_ACCESS_KEY") or merged_env.get("R2_SECRET_ACCESS_KEY", ""),
        session_token=merged_env.get("AWS_SESSION_TOKEN") or merged_env.get("R2_SESSION_TOKEN"),
        region=merged_env.get("AWS_DEFAULT_REGION") or merged_env.get("AWS_REGION"),
    )
    merged_env.setdefault("R2_PROFILE", r2_profile)

    # Disable torch.compile / TorchDynamo for eval. The moving ``:latest`` base image's
    # torch (2.12 + triton 3.7) ships a broken ``torch._inductor`` import, and
    # ``olmo_core.generate.utils`` applies ``@torch.compile`` at import time, so importing
    # the generation module crashes the container. Generation correctness does not need
    # compilation (we recompute the forward each step with use_cache=False), so disabling
    # dynamo sidesteps the broken inductor path entirely.
    merged_env["TORCH_COMPILE_DISABLE"] = "1"
    merged_env["TORCHDYNAMO_DISABLE"] = "1"

    out_path = f"/tmp/long_context_{label.replace('/', '_')}.json"
    cmd = [
        sys.executable, "-m", "scripts.eval.long_context.run_long_context",
        "--checkpoint", checkpoint_uri,
        "--label", label,
        "--output", out_path,
        "--num-samples", str(cli["num_samples"]),
        "--batch-size", str(cli["batch_size"]),
        "--max-new-tokens", str(cli["max_new_tokens"]),
        "--dump-generations", str(cli.get("dump_generations", 0)),
        "--dtype", cli["dtype"],
        "--context-lengths", *[str(x) for x in cli["context_lengths"]],
        "--depths", *[str(x) for x in cli["depths"]],
        "--tasks", *cli["tasks"],
    ]
    if cli.get("attention_backend"):
        cmd += ["--attention-backend", cli["attention_backend"]]
    print(f"[Modal] {' '.join(cmd)}")
    sys.stdout.flush()
    subprocess.check_call(cmd, cwd=str(REMOTE_LAUNCH_DIR / "src"), env=merged_env)

    with open(out_path) as f:
        return json.load(f)


@app.local_entrypoint()
def main(
    checkpoint_uri: str,
    labels: str = "",
    context_lengths: str = "1024,2048,4096,8192,16384,32768",
    depths: str = "0.0,0.25,0.5,0.75,1.0",
    tasks: str = "niah_single,niah_multikey,niah_multivalue,variable_tracking",
    num_samples: int = 20,
    batch_size: int = 4,
    max_new_tokens: int = 48,
    dtype: str = "bfloat16",
    attention_backend: str = "",
    gpu_type: str = "H100",
    output_dir: str = "./long_context_results",
    dump_generations: int = 0,
):
    """Fan out one eval container per checkpoint and collect the result JSONs."""
    ckpts = [c.strip() for c in checkpoint_uri.split(",") if c.strip()]
    label_list = [l.strip() for l in labels.split(",") if l.strip()]
    if not label_list:
        label_list = [Path(c.rstrip("/")).name for c in ckpts]
    assert len(label_list) == len(ckpts), "labels count must match checkpoints count"

    cli = dict(
        context_lengths=[int(x) for x in context_lengths.split(",")],
        depths=[float(x) for x in depths.split(",")],
        tasks=[t.strip() for t in tasks.split(",")],
        num_samples=num_samples, batch_size=batch_size, max_new_tokens=max_new_tokens,
        dtype=dtype, attention_backend=(attention_backend or None),
        dump_generations=dump_generations,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # starmap fans out across containers in parallel.
    args = [(ckpt, label, cli) for ckpt, label in zip(ckpts, label_list)]
    for (ckpt, label, _), result in zip(args, run_eval.starmap(args)):
        out = Path(output_dir) / f"long_context_{label.replace('/', '_')}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"[local] wrote {out}")


if __name__ == "__main__":
    # Allow `python long_context_modal.py ...` to print usage when not under `modal run`.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.parse_known_args()
    print("Launch with:  modal run long_context_modal.py --checkpoint-uri <uri> [...]",
          file=sys.stderr)

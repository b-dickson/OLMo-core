#!/usr/bin/env python3
"""Modal credential smoke test.

Submits a tiny function to Modal on a single T4 GPU (cheapest tier) using the
same base image and ``r2-creds`` secret as the attention-ladder launcher, and
reports whether each credential channel is wired up:

  - GPU resource allocation (torch.cuda visibility)
  - Modal secret materialization (expected env vars present)
  - R2 read access via boto3 (list a few keys in the ``llm-data`` bucket)
  - Eval-data mount at /data/eval-data
  - W&B authentication (wandb.Api().viewer)
  - Hugging Face token (huggingface_hub whoami)

Run with::

    modal run src/scripts/modal/smoke_test.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

MODAL_SECRET_NAME = "r2-creds"
R2_BUCKET = "llm-data"
DOCKER_IMAGE = os.environ.get("OLMO_DOCKER_IMAGE", "ghcr.io/allenai/olmo-core:latest")

# Mirror the eval-data mount from attention_ladder_modal.py so this smoke test
# verifies the same mount path used by the real launcher. Inside the Modal
# container, __file__ is /root/smoke_test.py and parents[3] would IndexError —
# we only need REPO_ROOT/LOCAL_EVAL_DATA_DIR on the local side anyway.
try:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    LOCAL_EVAL_DATA_DIR = REPO_ROOT.parent / "datasets" / "eval-data"
except IndexError:
    REPO_ROOT = Path("/repo/olmo-core")
    LOCAL_EVAL_DATA_DIR = Path("/nonexistent")  # not used inside container
REMOTE_EVAL_DATA_PATH = "/data/eval-data"

app = modal.App("olmo-modal-smoke-test")

image = modal.Image.from_registry(DOCKER_IMAGE, add_python="3.12").pip_install(
    "boto3", "wandb", "huggingface_hub"
)
if LOCAL_EVAL_DATA_DIR.exists():
    image = image.add_local_dir(str(LOCAL_EVAL_DATA_DIR), remote_path=REMOTE_EVAL_DATA_PATH)


@app.function(
    gpu="T4:1",
    image=image,
    timeout=600,
    retries=0,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
)
def smoke_test() -> dict:
    results: dict = {}

    print("=== GPU check ===")
    try:
        import torch

        results["gpu_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results["gpu_count"] = torch.cuda.device_count()
            results["gpu_name"] = torch.cuda.get_device_name(0)
            print(f"  CUDA available: True")
            print(f"  GPU count: {results['gpu_count']}")
            print(f"  GPU name: {results['gpu_name']}")
        else:
            print("  CUDA available: False")
    except Exception as e:
        results["gpu_error"] = str(e)
        print(f"  ERROR: {e}")

    print("\n=== Secret env vars ===")
    expected_keys = [
        "R2_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "R2_PROFILE",
        "WANDB_API_KEY",
        "HF_TOKEN",
    ]
    secret_status = {}
    for key in expected_keys:
        val = os.environ.get(key)
        present = bool(val)
        secret_status[key] = present
        # AWS_PROFILE / R2_PROFILE / region values are not sensitive — show them.
        if present and key in {"AWS_PROFILE", "R2_PROFILE", "AWS_DEFAULT_REGION"}:
            print(f"  {key}: present (={val})")
        else:
            print(f"  {key}: {'present' if present else 'MISSING'}")
    results["secrets"] = secret_status

    print("\n=== R2 access (boto3, explicit credentials) ===")
    try:
        import boto3

        # Isolate this from any AWS_PROFILE the secret sets — pass an explicit
        # Session with credentials so boto3 does not try to look up a profile.
        session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION", "auto"),
            profile_name=None,
        )
        s3 = session.client("s3", endpoint_url=os.environ.get("R2_ENDPOINT_URL"))
        resp = s3.list_objects_v2(Bucket=R2_BUCKET, MaxKeys=3)
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        results["r2_explicit_ok"] = True
        results["r2_sample_keys"] = keys
        print(f"  Bucket '{R2_BUCKET}' listed successfully")
        print(f"  Sample keys (up to 3): {keys}")
    except Exception as e:
        results["r2_explicit_ok"] = False
        results["r2_explicit_error"] = str(e)
        print(f"  ERROR: {e}")

    print("\n=== R2 access (boto3, via AWS profile, mimics launcher) ===")
    # Mirror the launcher: write ~/.aws/{config,credentials} for the configured
    # profile, then let boto3 resolve credentials via that profile.
    try:
        import boto3

        profile = os.environ.get("R2_PROFILE") or os.environ.get("AWS_PROFILE") or "r2"
        endpoint = os.environ.get("R2_ENDPOINT_URL", "")
        region = os.environ.get("AWS_DEFAULT_REGION") or "auto"
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

        aws_dir = Path.home() / ".aws"
        aws_dir.mkdir(parents=True, exist_ok=True)
        (aws_dir / "config").write_text(
            f"[profile {profile}]\nregion = {region}\nendpoint_url = {endpoint}\n"
        )
        (aws_dir / "credentials").write_text(
            f"[{profile}]\naws_access_key_id = {access_key}\n"
            f"aws_secret_access_key = {secret_key}\n"
        )

        session = boto3.Session(profile_name=profile)
        s3 = session.client("s3", endpoint_url=endpoint)
        resp = s3.list_objects_v2(Bucket=R2_BUCKET, MaxKeys=3)
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        results["r2_profile_ok"] = True
        print(f"  Profile '{profile}' resolved; bucket '{R2_BUCKET}' listed successfully")
        print(f"  Sample keys (up to 3): {keys}")
    except Exception as e:
        results["r2_profile_ok"] = False
        results["r2_profile_error"] = str(e)
        print(f"  ERROR: {e}")

    print("\n=== Eval data mount ===")
    import glob as _glob

    if os.path.isdir(REMOTE_EVAL_DATA_PATH):
        npy_files = sorted(_glob.glob(f"{REMOTE_EVAL_DATA_PATH}/**/*.npy", recursive=True))
        results["eval_mount_ok"] = True
        results["eval_file_count"] = len(npy_files)
        print(f"  {REMOTE_EVAL_DATA_PATH} exists with {len(npy_files)} .npy file(s)")
        for f in npy_files[:5]:
            print(f"    {f}")
        if len(npy_files) > 5:
            print(f"    ... and {len(npy_files) - 5} more")
    else:
        results["eval_mount_ok"] = False
        print(f"  {REMOTE_EVAL_DATA_PATH} does NOT exist inside container!")

    print("\n=== W&B login validation ===")
    try:
        import wandb

        api = wandb.Api()
        viewer = api.viewer
        username = (
            viewer.get("username", "<unknown>") if isinstance(viewer, dict) else str(viewer)
        )
        results["wandb_ok"] = True
        results["wandb_user"] = username
        print(f"  W&B authenticated as: {username}")
    except Exception as e:
        results["wandb_ok"] = False
        results["wandb_error"] = str(e)
        print(f"  ERROR: {e}")

    print("\n=== HF token validation ===")
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        info = api.whoami()
        results["hf_ok"] = True
        results["hf_user"] = info.get("name", "<unknown>")
        print(f"  HF authenticated as: {results['hf_user']}")
    except Exception as e:
        results["hf_ok"] = False
        results["hf_error"] = str(e)
        print(f"  ERROR: {e}")

    print("\n=== Summary ===")
    checks = [
        ("GPU", results.get("gpu_available", False)),
        ("R2 (explicit creds)", results.get("r2_explicit_ok", False)),
        ("R2 (profile, launcher path)", results.get("r2_profile_ok", False)),
        ("Eval data mount", results.get("eval_mount_ok", False)),
        ("W&B", results.get("wandb_ok", False)),
        ("HF", results.get("hf_ok", False)),
    ]
    for name, ok in checks:
        print(f"  {name}: {'OK' if ok else 'FAIL'}")

    return results


@app.local_entrypoint()
def main() -> None:
    print("Submitting smoke test to Modal (T4:1)...")
    results = smoke_test.remote()
    print("\n=== Local: results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

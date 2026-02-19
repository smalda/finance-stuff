#!/usr/bin/env python3
"""Remote GPU execution via Kaggle API.

Runs a Python script on a free Kaggle GPU. Packages the week's code/
directory, uploads it, runs the script, and downloads results back.
The target script is unaware of Kaggle — it sees --device cuda normally.

Usage:
    python3 course/shared/kaggle_gpu_runner.py <script_path> [script_args...]

Prerequisites:
    pip install kaggle
    Place API token at ~/.kaggle/kaggle.json
"""

import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

# Packages from pyproject.toml that aren't pre-installed on Kaggle.
# Update this list as the project evolves.
_EXTRA_DEPS = [
    "arch", "linearmodels", "fredapi", "cvxpy", "hmmlearn",
    "optuna", "polars", "pyarrow", "dowhy",
    "stable-baselines3", "gymnasium", "sentence-transformers",
]

_POLL_INTERVAL = 15  # seconds between status checks
_DATASET_WAIT = 300  # max seconds to wait for dataset processing
_KERNEL_TIMEOUT = 3600  # max seconds to wait for kernel completion


def _read_username() -> str:
    """Read Kaggle username from ~/.kaggle/kaggle.json."""
    import os
    if u := os.environ.get("KAGGLE_USERNAME"):
        return u
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    print("Error: set KAGGLE_USERNAME or place token at ~/.kaggle/kaggle.json",
          file=sys.stderr)
    sys.exit(1)


def _resolve_week_dir(script_path: Path) -> Path:
    """Find the weekNN_* directory from a script path."""
    for parent in script_path.parents:
        if parent.name.startswith("week") and parent.parent.name == "course":
            return parent
    print(f"Error: cannot find weekNN_* directory from {script_path}",
          file=sys.stderr)
    sys.exit(1)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, printing stderr on failure."""
    r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if r.returncode != 0 and r.stderr:
        print(r.stderr, file=sys.stderr)
    return r


# ── Package ──────────────────────────────────────────────────────────

def _package_dataset(week_dir: Path, staging: Path, username: str) -> str:
    """Copy week's code/ to staging and create dataset metadata."""
    data_dir = staging / "data"
    dest = data_dir / "course" / week_dir.name / "code"
    shutil.copytree(
        week_dir / "code", dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "logs"),
    )

    slug = f"finance-{week_dir.name}-gpu"
    meta = {
        "title": slug,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (data_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    return f"{username}/{slug}"


def _make_bootstrap(script_path: Path, week_dir: Path,
                    extra_args: list[str], dataset_slug: str) -> str:
    """Generate the bootstrap script that runs on Kaggle."""
    week_name = week_dir.name
    rel_script = str(script_path.relative_to(week_dir / "code"))
    deps = " ".join(_EXTRA_DEPS)
    dataset_name = dataset_slug.split("/")[1]

    return f'''#!/usr/bin/env python3
"""Kaggle bootstrap — auto-generated, do not edit."""
import shutil, subprocess, sys, tarfile
from pathlib import Path

INPUT = Path("/kaggle/input/{dataset_name}")
WORK = Path("/kaggle/working")
WEEK = WORK / "course" / "{week_name}" / "code"

# 1. Reconstruct directory tree
shutil.copytree(INPUT / "course", WORK / "course", dirs_exist_ok=True)

# 2. Install potentially missing deps (pre-installed ones are skipped quickly)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", {" ".join(repr(d) for d in _EXTRA_DEPS)}],
    check=False,
)

# 3. Run the target script
script = WEEK / "{rel_script}"
result = subprocess.run(
    [sys.executable, str(script)] + {extra_args!r},
    cwd=str(WORK),
)

# 4. Package output for download (tar avoids kaggle output dir issues)
tar_path = WORK / "results.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    logs = WEEK / "logs"
    cache = WEEK / ".cache"
    if logs.exists():
        tar.add(logs, arcname="code/logs")
    if cache.exists():
        tar.add(cache, arcname="code/.cache")

sys.exit(result.returncode)
'''


# ── Kaggle API calls ─────────────────────────────────────────────────

def _upload_dataset(staging: Path, dataset_slug: str):
    """Create or update the Kaggle dataset and wait until ready."""
    data_dir = staging / "data"

    # Check if dataset exists
    r = _run(["kaggle", "datasets", "status", dataset_slug])
    if r.returncode == 0:
        print("  Updating existing dataset...", flush=True)
        _run(["kaggle", "datasets", "version", "-p", str(data_dir),
              "-m", "update", "--dir-mode", "tar"])
    else:
        print("  Creating new dataset...", flush=True)
        _run(["kaggle", "datasets", "create", "-p", str(data_dir)])

    # Wait for dataset to be ready
    start = time.time()
    while time.time() - start < _DATASET_WAIT:
        r = _run(["kaggle", "datasets", "status", dataset_slug])
        if "ready" in r.stdout.lower():
            return
        time.sleep(5)
    print("Error: dataset not ready after timeout", file=sys.stderr)
    sys.exit(1)


def _push_kernel(staging: Path, bootstrap: str,
                 dataset_slug: str, username: str, week_name: str) -> str:
    """Write bootstrap + metadata, push kernel."""
    kernel_dir = staging / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "bootstrap.py").write_text(bootstrap)

    slug = f"finance-{week_name}-gpu-run"
    meta = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": "bootstrap.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [dataset_slug],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    r = _run(["kaggle", "kernels", "push", "-p", str(kernel_dir)])
    if r.returncode != 0:
        print(f"Error pushing kernel: {r.stdout}", file=sys.stderr)
        sys.exit(1)
    return f"{username}/{slug}"


def _poll_kernel(kernel_slug: str) -> bool:
    """Poll until kernel completes or fails."""
    start = time.time()
    while time.time() - start < _KERNEL_TIMEOUT:
        r = _run(["kaggle", "kernels", "status", kernel_slug])
        status = r.stdout.strip().lower()
        if "complete" in status:
            return True
        if "error" in status or "cancel" in status:
            print(f"  Kernel failed: {r.stdout.strip()}", file=sys.stderr)
            return False
        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] status: {r.stdout.strip()}", flush=True)
        time.sleep(_POLL_INTERVAL)
    print("Error: kernel timed out", file=sys.stderr)
    return False


def _download_and_unpack(kernel_slug: str, week_dir: Path):
    """Download kernel output and unpack results to local paths."""
    with tempfile.TemporaryDirectory() as dl_dir:
        dl = Path(dl_dir)
        _run(["kaggle", "kernels", "output", kernel_slug, "-p", str(dl)])

        tar_path = dl / "results.tar.gz"
        if tar_path.exists():
            with tarfile.open(tar_path) as tar:
                tar.extractall(week_dir)
            print("  Results unpacked to local paths.", flush=True)
        else:
            print("  Warning: no results.tar.gz in output.", file=sys.stderr)
            # List what was downloaded for debugging
            for f in dl.iterdir():
                print(f"    {f.name}", file=sys.stderr)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 course/shared/kaggle_gpu_runner.py "
              "<script_path> [script_args...]", file=sys.stderr)
        sys.exit(1)

    script_path = Path(sys.argv[1]).resolve()
    extra_args = sys.argv[2:]

    if not script_path.exists():
        print(f"Error: script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    username = _read_username()
    week_dir = _resolve_week_dir(script_path)
    week_name = week_dir.name

    print(f"=== Kaggle GPU Runner ===", flush=True)
    print(f"  Script: {script_path.relative_to(week_dir.parent.parent)}",
          flush=True)
    print(f"  Week:   {week_name}", flush=True)

    with tempfile.TemporaryDirectory(prefix="kaggle_gpu_") as tmpdir:
        staging = Path(tmpdir)

        # 1. Package
        print("1/5 Packaging dataset...", flush=True)
        dataset_slug = _package_dataset(week_dir, staging, username)

        # 2. Upload
        print("2/5 Uploading dataset...", flush=True)
        _upload_dataset(staging, dataset_slug)

        # 3. Generate bootstrap + push kernel
        print("3/5 Pushing kernel...", flush=True)
        bootstrap = _make_bootstrap(script_path, week_dir,
                                    extra_args, dataset_slug)
        kernel_slug = _push_kernel(staging, bootstrap, dataset_slug,
                                   username, week_name)

        # 4. Poll
        print("4/5 Waiting for kernel...", flush=True)
        success = _poll_kernel(kernel_slug)

        # 5. Download
        print("5/5 Downloading results...", flush=True)
        _download_and_unpack(kernel_slug, week_dir)

        if not success:
            print("\nKernel failed. Check output above for errors.",
                  file=sys.stderr)
            sys.exit(1)

    print("=== Done ===", flush=True)


if __name__ == "__main__":
    main()

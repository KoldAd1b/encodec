import json
import os
import platform
import random
import socket
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def make_json_safe(value):
    if is_dataclass(value):
        return make_json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(make_json_safe(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def _run_command(command):
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except OSError as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def collect_environment_snapshot():
    packages = {}
    for name in [
        "torch",
        "torchaudio",
        "torchcodec",
        "accelerate",
        "transformers",
        "librosa",
        "soundfile",
        "yaml",
        "wandb",
        "triton",
    ]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            packages[name] = f"unavailable: {type(exc).__name__}: {exc}"

    cuda = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cuda["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "major": props.major,
                    "minor": props.minor,
                }
            )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "command": sys.argv,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "packages": packages,
        "cuda": cuda,
        "ffmpeg": _run_command(["ffmpeg", "-version"])["stdout"].splitlines()[:1],
    }


def collect_git_snapshot():
    return {
        "commit": _run_command(["git", "rev-parse", "HEAD"])["stdout"],
        "branch": _run_command(["git", "branch", "--show-current"])["stdout"],
        "status_short": _run_command(["git", "status", "--short"])["stdout"],
        "diff": _run_command(["git", "diff", "--", "."])["stdout"],
    }


def save_run_snapshot(path_to_experiment, run_config):
    path_to_experiment = Path(path_to_experiment)
    path_to_experiment.mkdir(parents=True, exist_ok=True)
    git_snapshot = collect_git_snapshot()

    write_json(path_to_experiment / "config_snapshot.json", run_config)
    write_json(path_to_experiment / "environment_snapshot.json", collect_environment_snapshot())
    write_json(
        path_to_experiment / "run_metadata.json",
        {
            "git_commit": git_snapshot["commit"],
            "git_branch": git_snapshot["branch"],
            "git_status_short": git_snapshot["status_short"],
        },
    )
    (path_to_experiment / "git_diff.patch").write_text(git_snapshot["diff"])


def _safe_audio_info(path):
    try:
        info = sf.info(path)
        return {
            "sample_rate": int(info.sample_rate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "duration": float(info.frames / info.sample_rate) if info.sample_rate else None,
        }
    except Exception as sf_exc:
        try:
            waveform, sample_rate = torchaudio.load(path)
            return {
                "sample_rate": int(sample_rate),
                "channels": int(waveform.shape[0]),
                "frames": int(waveform.shape[-1]),
                "duration": float(waveform.shape[-1] / sample_rate) if sample_rate else None,
            }
        except Exception as torch_exc:
            return {
                "error": (
                    f"soundfile {type(sf_exc).__name__}: {sf_exc}; "
                    f"torchaudio {type(torch_exc).__name__}: {torch_exc}"
                )
            }


def _percentiles(values):
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def build_dataset_report(
    train_files,
    test_files,
    sampling_rate,
    segment_length,
    max_audio_files=None,
    seed=None,
):
    def summarize_split(name, paths):
        paths = list(paths)
        existing = [p for p in paths if Path(p).is_file()]
        missing = [p for p in paths if not Path(p).is_file()]
        sample_paths = existing
        if max_audio_files is not None and len(existing) > max_audio_files:
            rng = random.Random(seed)
            sample_paths = rng.sample(existing, max_audio_files)

        durations = []
        sample_rates = Counter()
        channels = Counter()
        corrupt = []
        too_short = 0
        min_duration = segment_length / sampling_rate

        for path in sample_paths:
            info = _safe_audio_info(path)
            if "error" in info:
                corrupt.append({"path": path, "error": info["error"]})
                continue
            duration = info["duration"]
            durations.append(duration)
            sample_rates[str(info["sample_rate"])] += 1
            channels[str(info["channels"])] += 1
            if duration is not None and duration < min_duration:
                too_short += 1

        return {
            "name": name,
            "num_files": len(paths),
            "missing_count": len(missing),
            "missing_examples": missing[:20],
            "extension_counts": dict(Counter(Path(p).suffix.lower() for p in paths)),
            "profiled_count": len(sample_paths),
            "corrupt_count": len(corrupt),
            "corrupt_examples": corrupt[:20],
            "sample_rate_counts": dict(sample_rates),
            "channel_counts": dict(channels),
            "duration_seconds": _percentiles(durations),
            "profiled_hours": float(sum(durations) / 3600.0) if durations else 0.0,
            "too_short_for_segment_count": too_short,
        }

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_rate": sampling_rate,
        "segment_length": segment_length,
        "segment_duration_seconds": segment_length / sampling_rate,
        "max_audio_files_per_split": max_audio_files,
        "splits": {
            "train": summarize_split("train", train_files),
            "test": summarize_split("test", test_files),
        },
    }


def save_reconstruction_spectrogram(path, original, reconstruction, sampling_rate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original = original.detach().cpu().squeeze().float().numpy()
    reconstruction = reconstruction.detach().cpu().squeeze().float().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].specgram(original, Fs=sampling_rate, NFFT=1024, noverlap=768)
    axes[0].set_title("Original")
    axes[0].set_ylabel("Frequency")
    axes[1].specgram(reconstruction, Fs=sampling_rate, NFFT=1024, noverlap=768)
    axes[1].set_title("Reconstruction")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

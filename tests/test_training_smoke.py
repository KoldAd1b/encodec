import soundfile as sf
import torch

from modules.encodec import EnCodecConfig, EncodecModel
from train import EncodecTrainer


def _write_manifest(tmp_path, name, count=2, sample_rate=24000, length=4096):
    paths = []
    for idx in range(count):
        path = tmp_path / f"{name}_{idx}.wav"
        audio = torch.randn(length).mul(0.01).numpy()
        sf.write(path, audio, sample_rate)
        paths.append(path)

    manifest = tmp_path / f"{name}.txt"
    manifest.write_text("\n".join(str(path) for path in paths) + "\n")
    return manifest


def _tiny_config():
    return EnCodecConfig(
        channels=1,
        dimension=8,
        n_filters=4,
        n_residual_layers=1,
        ratios=(2, 2),
        activation="ELU",
        norm="none",
        kernel_size=3,
        last_kernel_size=3,
        residual_kernel_size=3,
        lstm=0,
        num_quantizers=2,
        codebook_size=8,
        kmeans_init=True,
        kmeans_iters=1,
        threshold_ema_dead_code=0,
    )


def test_one_step_training_writes_observability_files(tmp_path):
    train_manifest = _write_manifest(tmp_path, "train")
    test_manifest = _write_manifest(tmp_path, "test")

    model = EncodecModel(_tiny_config())
    training_config = {
        "experiment_name": "smoke",
        "working_directory": str(tmp_path / "runs"),
        "path_to_train_manifest": str(train_manifest),
        "path_to_test_manifest": str(test_manifest),
        "sampling_rate": 24000,
        "segment_length": 4096,
        "num_mels": 16,
        "total_iterations": 1,
        "warmup_iterations": 0,
        "checkpoint_iterations": 1000,
        "eval_iterations": 1000,
        "generation_iterations": 1000,
        "per_gpu_batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        "console_out_iters": 1,
        "log_wandb": False,
        "use_balancer": False,
        "dataset_report_max_files": 2,
        "num_samples_for_reconstruction": 1,
        "save_reconstruction_spectrograms": False,
    }

    trainer = EncodecTrainer(model, discriminator=None, training_config=training_config)
    trainer.train()

    run_dir = tmp_path / "runs" / "smoke"
    assert (run_dir / "config_snapshot.json").exists()
    assert (run_dir / "environment_snapshot.json").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "git_diff.patch").exists()
    assert (run_dir / "dataset_report.json").exists()
    assert (run_dir / "final_checkpoint").exists()

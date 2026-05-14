import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
import yaml

from modules.encodec import EncodecModel, EnCodecConfig
from utils import save_audios

warnings.filterwarnings("ignore")


def load_yaml(path_to_yaml):
    with open(path_to_yaml, "r") as file:
        return yaml.safe_load(file)


def first_manifest_path(path_to_manifest):
    with open(path_to_manifest, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                return line
    raise ValueError(f"No audio paths found in {path_to_manifest}")


def load_audio(path_to_audio, sr):
    waveform, audio_sr = torchaudio.load(path_to_audio)

    if audio_sr != sr:
        waveform = torchaudio.transforms.Resample(audio_sr, sr)(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.unsqueeze(0)


def compute_mel(audio, sr, n_fft=1024, hop_length=256, n_mels=80):
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0,
    )
    mel = mel_transform(audio)
    return torch.log(mel + 1e-5)


def plot_mel_comparison(original, reconstruction, sr, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mel_orig = compute_mel(original.squeeze(0), sr)
    mel_recon = compute_mel(reconstruction.detach().cpu().squeeze(0), sr)

    mel_orig = mel_orig.squeeze(0).numpy()
    mel_recon = mel_recon.squeeze(0).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    im0 = axes[0].imshow(mel_orig, aspect="auto", origin="lower")
    axes[0].set_title("Original Audio Mel Spectrogram")
    axes[0].set_ylabel("Mel bins")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mel_recon, aspect="auto", origin="lower")
    axes[1].set_title("Reconstructed Audio Mel Spectrogram")
    axes[1].set_ylabel("Mel bins")
    axes[1].set_xlabel("Time")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct audio with a trained EnCodec checkpoint.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--checkpoint",
        default="dir/EnCodecTrainerLibriTTS/checkpoint_190000/pytorch_model.bin",
    )
    parser.add_argument("--audio", default=None, help="Audio file to reconstruct. Defaults to first data/test.txt entry.")
    parser.add_argument("--manifest", default="data/test.txt", help="Manifest used when --audio is omitted.")
    parser.add_argument("--output", default="dir/EnCodecTrainerLibriTTS/manual_reconstruction.wav")
    parser.add_argument("--spectrogram", default="dir/EnCodecTrainerLibriTTS/manual_reconstruction.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    audio_path = args.audio or first_manifest_path(args.manifest)
    sample_rate = config["training_config"]["sampling_rate"]

    model = EncodecModel(EnCodecConfig(**config["generator_config"])).to(args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    audio = load_audio(audio_path, sample_rate).to(args.device)

    with torch.no_grad():
        tokens, scale = model.tokenize(audio)
        reconstruction = model.decode(tokens, scale, max_len=audio.shape[-1])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_audios(reconstruction.cpu(), str(output_path), sample_rate)
    plot_mel_comparison(audio.cpu(), reconstruction.cpu(), sample_rate, args.spectrogram)

    print(f"audio: {audio_path}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"tokens shape: {tuple(tokens.shape)}")
    print(f"reconstruction: {output_path}")
    print(f"spectrogram: {args.spectrogram}")


if __name__ == "__main__":
    main()

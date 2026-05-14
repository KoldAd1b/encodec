import argparse
import warnings

import torch
import torchaudio
import yaml

from modules.encodec import EncodecModel, EnCodecConfig

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


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize audio with a trained EnCodec checkpoint.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--checkpoint",
        default="dir/EnCodecTrainerLibriTTS/checkpoint_190000/pytorch_model.bin",
    )
    parser.add_argument("--audio", default=None, help="Audio file to tokenize. Defaults to first data/test.txt entry.")
    parser.add_argument("--manifest", default="data/test.txt", help="Manifest used when --audio is omitted.")
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

    print(f"audio: {audio_path}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"tokens shape: {tuple(tokens.shape)}")
    print(tokens.cpu())
    print(f"scale shape: {tuple(scale.shape)}")


if __name__ == "__main__":
    main()

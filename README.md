# EnCodec

This project is an implementation of an EnCodec-style neural audio codec. The goal is to train a model that can compress waveform audio into discrete latent codes and reconstruct high-quality audio from those codes.

EnCodec is useful because the compressed representation is much smaller than raw audio while still preserving enough information for perceptually convincing reconstruction. Those discrete codes can also be used as audio tokens for downstream generative audio models.

## Overview

The model follows the standard neural codec structure:

```text
audio waveform
  -> encoder
  -> residual vector quantizer
  -> decoder
  -> reconstructed waveform
```

The encoder converts raw audio into a lower-rate latent sequence. The quantizer maps that latent sequence into discrete codebook entries. The decoder converts the quantized representation back into waveform audio.

## Main Components

The project is organized around a few core model components:

- **Encoder**: a stack of 1D convolutional blocks that downsamples the waveform into latent features.
- **Sequence modeling**: optional recurrent or temporal layers that improve context modeling inside the latent representation.
- **Residual vector quantizer**: a sequence of vector quantizers that progressively encode the residual error between the encoder output and the quantized approximation.
- **Decoder**: a stack of transposed convolutional blocks that upsamples quantized latents back to waveform audio.
- **Losses**: reconstruction, spectral, commitment, and optionally adversarial losses for improving perceptual quality.
- **Training loop**: dataset loading, audio preprocessing, distributed training, checkpointing, validation, and sample export.

## Training Objective

The model should learn to reconstruct audio while keeping the bottleneck discrete and compact. A typical training objective combines:

- Waveform reconstruction loss.
- Multi-scale spectral reconstruction loss.
- Quantizer commitment loss.
- Optional adversarial loss from audio discriminators.
- Optional feature matching loss from discriminator activations.

The reconstruction-only objective should work first. Adversarial training should be added only after the autoencoder path is stable, because discriminator losses make debugging much harder.

## Quantization

The quantizer uses residual vector quantization. Instead of relying on one codebook to represent the full latent vector, multiple quantizers are applied in sequence. Each quantizer encodes the residual left by the previous one:

```text
residual_0 = encoder_output
code_0 = quantizer_0(residual_0)
residual_1 = residual_0 - decode(code_0)
code_1 = quantizer_1(residual_1)
...
```

The final quantized representation is the sum of all selected codebook vectors. This makes it possible to trade bitrate against quality by changing the number of active quantizers.

## Dataset Expectations

Training data should be a collection of audio files. The training pipeline should handle:

- Loading common audio formats.
- Resampling to the configured sample rate.
- Converting to mono or preserving channels, depending on the model target.
- Random cropping fixed-length segments.
- Normalizing audio levels consistently.
- Skipping unreadable or too-short files.

Validation should export reconstructed audio examples so model quality can be checked by listening, not only by scalar losses.

## Development Priorities

The first milestone is a minimal end-to-end model:

```text
audio batch -> encoder -> quantizer -> decoder -> reconstruction loss
```

Recommended order:

1. Add project dependencies and a repeatable environment setup.
2. Add smoke tests for tensor shapes and encode/decode round trips.
3. Implement the encoder and decoder.
4. Add a top-level model class that wires the encoder, quantizer, and decoder together.
5. Add a small single-device training loop with waveform and spectral reconstruction losses.
6. Add checkpoint save/resume.
7. Add validation and reconstructed audio export.
8. Add distributed training support.
9. Add adversarial training once reconstruction-only training is stable.

## Dependencies

Core dependencies are expected to include:

```text
torch
torchaudio
einops
accelerate
numpy
tqdm
soundfile
```

Useful optional dependencies:

```text
auraloss
tensorboard
wandb
matplotlib
pytest
```

The exact dependency list should be captured in a `requirements.txt`, `environment.yml`, or `pyproject.toml` once the training stack is finalized.

## Training

The training entrypoint should eventually support configuration-driven runs, for example:

```bash
python train.py --config configs/encodec.yaml
```

For distributed training, the project can use either Hugging Face Accelerate or native PyTorch distributed launch. The final choice should be reflected in the training scripts and configuration files.

## Evaluation

Evaluation should include both objective and subjective checks:

- Reconstruction loss on held-out audio.
- Spectral loss on held-out audio.
- Codebook usage statistics.
- Reconstructed audio samples.
- Compression bitrate for a chosen number of quantizers.

Listening tests matter for this project. Low loss does not always mean the reconstructed audio sounds good.

## References

- Meta AI, **High Fidelity Neural Audio Compression**
- Meta's EnCodec implementation and released model behavior
- SoundStream-style residual vector quantization
- Multi-scale STFT losses for neural audio generation

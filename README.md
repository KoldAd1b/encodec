# EnCodec From Scratch

This repository is an in-progress implementation of Meta's EnCodec-style neural audio codec. The goal is to build the model components, train them on local audio data, and scale training across 4 NVIDIA RTX 2080 Ti GPUs.

## Goal

EnCodec learns to compress raw audio into discrete codes and reconstruct the waveform from those codes. At a high level, the system should contain:

- A convolutional encoder that downsamples waveform audio into a latent sequence.
- A residual vector quantizer that converts latents into discrete codebook indices.
- A convolutional decoder that reconstructs audio from quantized latents.
- Reconstruction and adversarial losses for high-quality audio generation.
- A distributed training pipeline that can run efficiently on 4 GPUs.

The current repository contains early building blocks for the convolution stack, LSTM blocks, and residual vector quantization. It does not yet contain the complete EnCodec model or a training pipeline.

## Current Repository State

```text
modules/
  conv.py        Normalized 1D/2D convolution wrappers and strided audio conv helpers.
  lstm.py        Sequence LSTM block with optional bidirectional projection and skip connection.
  quantizer.py   Euclidean codebook, vector quantization, and residual vector quantization.
```

There is currently no package setup, dependency file, dataset loader, model assembly file, training script, checkpoint logic, evaluation script, or audio demo/inference entrypoint.

## Implemented Components

### Convolution Helpers

`modules/conv.py` defines:

- `NormConv1d`
- `NormConv2d`
- `NormTransposeConv1d`
- `NormTransposeConv2d`
- `SConv1d`
- `SConvTranspose1d`

These are intended to support EnCodec-style strided convolution and transpose convolution with optional normalization.

Known issue: `ConvLayerNorm.forward()` currently returns `None` instead of the normalized tensor. Any path using `norm="layer_norm"` will fail until this is fixed.

### LSTM Block

`modules/lstm.py` defines `SLSTM`, a sequence LSTM block that:

- Accepts tensors shaped `(batch, channels, length)`.
- Internally converts to `(length, batch, channels)` for PyTorch LSTM.
- Supports bidirectional LSTM output projection.
- Supports an optional skip connection.

This matches the kind of sequence modeling block often inserted between convolutional stages in neural audio codecs.

### Quantizer

`modules/quantizer.py` defines:

- K-means initialization helpers.
- An EMA-updated Euclidean codebook.
- `VectorQuantization`.
- `ResidualVectorQuantization`.

The quantizer includes early support for distributed training with Hugging Face Accelerate by gathering samples and reducing codebook statistics across processes.

## Target Architecture

The intended model should eventually look like:

```text
waveform
  -> encoder
  -> residual vector quantizer
  -> decoder
  -> reconstructed waveform
```

The training objective should combine:

- Time-domain reconstruction loss, such as L1 over waveform samples.
- Frequency-domain reconstruction loss, such as multi-scale STFT loss.
- Commitment loss from the vector quantizer.
- Optional adversarial loss with multi-scale or multi-period audio discriminators.
- Optional feature matching loss from discriminator intermediate activations.

## Training Target

The intended hardware target is 4x RTX 2080 Ti GPUs.

Important constraints for 2080 Ti training:

- Each GPU usually has 11 GB VRAM, so batch size and audio segment length need to be conservative.
- Mixed precision should be used where stable.
- Distributed data parallel training should be used through Accelerate or native PyTorch DDP.
- Gradient accumulation will probably be needed for larger effective batch sizes.
- Checkpointing should save model weights, optimizer state, scheduler state, quantizer buffers, and training step.

Recommended initial training setup:

```text
sample_rate: 24000 or 48000
segment_seconds: 1 to 3
per_gpu_batch_size: start with 2 to 4
gradient_accumulation_steps: tune based on memory
precision: fp16 or bf16 if supported
num_processes: 4
```

The 2080 Ti does not support bf16 acceleration like newer GPUs, so fp16 is the practical mixed-precision target.

## Missing Work

Before full training is possible, the project needs:

1. A dependency file, such as `requirements.txt` or `pyproject.toml`.
2. A complete encoder module.
3. A complete decoder module.
4. A top-level EnCodec model class that wires encoder, quantizer, and decoder together.
5. Dataset loading for audio files.
6. Audio preprocessing and random crop logic.
7. Training configuration.
8. Distributed training script.
9. Reconstruction losses.
10. Optional discriminator models and adversarial losses.
11. Checkpoint save/resume.
12. Validation loop.
13. Audio reconstruction export for listening tests.
14. Unit tests or smoke tests for tensor shapes and encode/decode round trips.

## Suggested Development Order

1. Fix module-level correctness issues.
2. Add a minimal dependency file.
3. Add unit tests for `SConv1d`, `SConvTranspose1d`, `SLSTM`, and `ResidualVectorQuantization`.
4. Implement the encoder and decoder.
5. Implement a top-level model forward pass.
6. Add a simple waveform reconstruction training loop on one GPU.
7. Add Accelerate configuration for 4 GPUs.
8. Add validation and audio sample export.
9. Add adversarial training after reconstruction-only training is stable.

## Local Environment

The current local environment used during inspection did not have `torch` installed, so runtime smoke tests could not be executed. At minimum, this project will need:

```text
torch
torchaudio
einops
accelerate
numpy
tqdm
soundfile or scipy
```

Optional dependencies will likely include:

```text
auraloss
tensorboard or wandb
matplotlib
pytest
```

## Example Launch Direction

Once a training script exists, the intended distributed launch should be similar to:

```bash
accelerate launch --num_processes 4 train.py --config configs/encodec.yaml
```

or:

```bash
torchrun --nproc_per_node=4 train.py --config configs/encodec.yaml
```

The exact command depends on whether the project standardizes on Accelerate or native PyTorch DDP.

## Current Status

This is not yet a trainable EnCodec implementation. It is a small prototype containing several low-level modules that can become part of the full model. The next milestone should be a minimal end-to-end autoencoder path:

```text
audio batch -> encoder -> quantizer -> decoder -> reconstruction loss
```

Once that works on a single GPU, distributed training across the 4x 2080 Ti setup can be added and validated.

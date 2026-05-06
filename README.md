# EnCodec

This repository implements an EnCodec-style neural audio codec in Python and
PyTorch. The model compresses waveform audio into discrete latent codes with
residual vector quantization, then reconstructs audio from those codes.

The compressed codes are much smaller than raw audio and can also be reused as
audio tokens for downstream generative audio models.

## Overview

The model follows the standard neural codec structure:

```text
audio waveform
  -> encoder
  -> residual vector quantizer
  -> decoder
  -> reconstructed waveform
```

The encoder converts raw audio into a lower-rate latent sequence. The quantizer
maps that latent sequence into discrete codebook entries. The decoder converts
the quantized representation back into waveform audio.

## Project Layout

- `modules/encodec.py`: top-level generator that wires the encoder, quantizer,
  and decoder.
- `modules/seanet.py`, `modules/conv.py`, `modules/lstm.py`, `modules/snake.py`:
  core encoder/decoder building blocks.
- `modules/quantizer.py`: residual vector quantization.
- `modules/discriminator.py`: adversarial audio discriminators.
- `loss.py`: reconstruction, spectral, feature matching, generator, and
  discriminator losses.
- `balancer.py`: gradient-balancing helper for multi-loss generator training.
- `dataset.py`: random segment audio dataset.
- `train.py`: `EncodecTrainer` implementation.
- `run.py`: configuration-driven training entrypoint.
- `audio_tokenize.py`: tokenization example.
- `inference.py`: reconstruction and mel comparison example.
- `build_db.py`: audio manifest builder.
- `pre.py`: optional long-audio chunking helper.
- `config/config.yaml`: runtime configuration.

## Data Preparation

Training uses text manifests by default. Each line should contain one path to an
audio file:

```text
/path/to/audio_0001.wav
/path/to/audio_0002.flac
/path/to/audio_0003.mp3
```

Create train and test manifests from an audio directory with:

```bash
python build_db.py /path/to/audio_root --split --train_ratio 0.9 \
  --train_file data/train.txt \
  --test_file data/test.txt
```

The checked-in config already points to those files:

```yaml
path_to_train_manifest: data/train.txt
path_to_test_manifest: data/test.txt
```

The dataset loader handles common preprocessing at training time:

- loads each path from the manifest
- resamples to `training_config.sampling_rate`
- averages multi-channel audio to mono
- randomly crops `training_config.segment_length`
- pads clips shorter than `segment_length`

If your source files are very long and expensive to load repeatedly, chunk them
first:

```bash
python pre.py /path/to/long_audio data/chunks --duration 15 --keep-remainder
python build_db.py data/chunks --split --train_ratio 0.9 \
  --train_file data/train.txt \
  --test_file data/test.txt
```

Short utterance datasets usually only need the manifest step.

## Training

Start a configured training run with:

```bash
python run.py --path_to_config config/config.yaml
```

For distributed training, launch the same entrypoint through Accelerate:

```bash
accelerate launch run.py --path_to_config config/config.yaml
```

Checkpoints, cached train/test splits, selected generation inputs, and generated
reconstructions are written under:

```text
dir/<experiment_name>/
```

The training loop supports:

- manifest or directory-based dataset construction
- cached train/test splits for resume consistency
- generator-only or adversarial training
- cosine learning-rate schedules with warmup
- optional gradient clipping
- optional loss balancing
- periodic validation
- periodic reconstructed sample export
- Accelerate checkpoint save and resume state

## Tokenization And Inference

`audio_tokenize.py` loads a configured model checkpoint and prints discrete
tokens for the first path in `data/test.txt`.

Verify paths before running it. The script currently contains explicit constants
for the config, checkpoint, and manifest locations.

```bash
python audio_tokenize.py
```

`inference.py` is a reconstruction example that tokenizes one audio file, decodes
it, saves reconstructed audio, and writes a mel-spectrogram comparison. It also
uses explicit local paths that should be edited before use.

```bash
python inference.py
```

## Validation

Run a quick syntax check across the project with:

```bash
python -m py_compile audio_tokenize.py dataset.py loss.py modules/*.py \
  balancer.py build_db.py inference.py pre.py run.py train.py utils.py
```

Check distributed tensor broadcast behavior with:

```bash
accelerate launch modules/gather_test.py
```

When tests are added, place them under `tests/` and run:

```bash
python -m pytest
```

Prioritize smoke tests for tensor shapes, encode/decode round trips, quantizer
codebook behavior, config construction, and CPU-only dataset loading.

## Dependencies

Core dependencies:

```text
torch
torchaudio
einops
accelerate
transformers
librosa
numpy
pyyaml
tqdm
soundfile
matplotlib
```

Optional development dependencies:

```text
pytest
wandb
```

No dependency lockfile is checked in yet. Capture the final environment in a
`requirements.txt`, `environment.yml`, or `pyproject.toml` once the training
stack is stable.

## References

- Meta AI, **High Fidelity Neural Audio Compression**
- Meta's EnCodec implementation and released model behavior
- SoundStream-style residual vector quantization
- Multi-scale STFT losses for neural audio generation

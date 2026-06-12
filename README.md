# livekit-plugins-hush

[LiveKit](https://livekit.io) noise suppression plugin using the [Hush](https://github.com/pulp-vision/Hush) speech enhancement model. Self-hosted, in-process, no cloud API.

Hush is built on [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) with an auxiliary separation head for background speaker suppression. Inference uses [DeepFilterLib](https://github.com/Rikorose/DeepFilterNet) for feature extraction and [ONNX Runtime](https://onnxruntime.ai) for the neural network. No PyTorch dependency at runtime.

---

## Installation

```
pip install livekit-plugins-hush
```

Dependencies: `livekit >= 1.0.25`, `livekit-agents >= 1.4.4`, `numpy >= 1.26.0`, `onnxruntime >= 1.17.0`, `deepfilterlib >= 0.5.4`.

---

## Usage

```python
from livekit.agents import room_io
from livekit.plugins import hush

await session.start(
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=hush.noise_suppression(),
        ),
    ),
)
```

One instance per session. The ONNX model is loaded once per worker process and shared across instances.

### Parameters

```python
hush.noise_suppression(
    strength=0.5,       # wet/dry blend: 0.0 = bypass, 1.0 = full (default 0.5)
    atten_lim_db=100.0, # max attenuation in dB (100.0 = unlimited)
    debug_logging=False, # log per-chunk RMS every 10 chunks
)
```

---

## Architecture

The model operates at 16 kHz with 10 ms frames (160 samples, 320-sample FFT). Processing is chunked: 32 frames (320 ms) are accumulated and processed together to provide the GRU layers with temporal context. The first chunk incurs 320 ms latency; subsequent chunks keep pace with the input stream.

### Signal flow

```
LiveKit AudioFrame (any rate, any channels)
  → resample to 16 kHz mono
  → buffer into 32-frame chunks
  → DeepFilterLib: STFT → ERB features + DF spectral features
  → ONNX: encoder → ERB decoder + DF decoder → enhanced spectrum
  → DeepFilterLib: ISTFT
  → wet/dry blend
  → upsample, restore channels
→ AudioFrame
```

### Model sharing

The ONNX session is a process-level singleton. Each `HushNoiseSuppressor` instance holds only per-stream DF state and audio buffers.

| Per worker | Per session |
|---|---|
| ~40 MB (onnxruntime + model) | ~1 MB (DF state + buffers) |

---

## Inference performance

| | Streaming (per chunk) | Batch (full file) |
|---|---|---|
| Chunk size | 32 frames (320 ms) | Full audio |
| Inference time | ~6.5 ms per chunk | ~8 ms per second of audio |
| Real-time factor | 0.02× | 0.008× |
| Throughput | 49× real-time (~150 chunks/sec) | 129× real-time |
| Model size | ~9 MB (3 ONNX files) | |

Measured on ARM64 Linux (aarch64). Steady-state throughput supports 100+ concurrent sessions per core.

---

## ONNX model

The ONNX model bundle is from the public PyTorch checkpoint ([weya-ai/hush](https://huggingface.co/weya-ai/hush)). Output parity is verified via `scripts/verify_against_pytorch.py`, which compares the ONNX pipeline output against the original PyTorch model.

---

## Audio samples

Noisy originals and their denoised counterparts. Two variants are provided:
- **Batch** (`hush-*.wav`) — full-file normalization, best quality. Matches the PyTorch `infer_single.py` pipeline.
- **Stream** (`hush-stream-*.wav`) — per-chunk normalization, real-time quality. Matches the LiveKit frame processor in production.

| Original | Batch | Stream |
|---|---|---|
| [`gym.wav`](docs/audio/originals/gym.wav) | [`hush-gym.wav`](docs/audio/hush-gym.wav) | [`hush-stream-gym.wav`](docs/audio/hush-stream-gym.wav) |
| [`krisp-original.wav`](docs/audio/originals/krisp-original.wav) | [`hush-krisp-original.wav`](docs/audio/hush-krisp-original.wav) | [`hush-stream-krisp-original.wav`](docs/audio/hush-stream-krisp-original.wav) |
| [`noproblem_raw.wav`](docs/audio/originals/noproblem_raw.wav) | [`hush-noproblem_raw.wav`](docs/audio/hush-noproblem_raw.wav) | [`hush-stream-noproblem_raw.wav`](docs/audio/hush-stream-noproblem_raw.wav) |
| [`taxi-sample.wav`](docs/audio/originals/taxi-sample.wav) | [`hush-taxi-sample.wav`](docs/audio/hush-taxi-sample.wav) | [`hush-stream-taxi-sample.wav`](docs/audio/hush-stream-taxi-sample.wav) |

---

## References

- [pulp-vision/Hush](https://github.com/pulp-vision/Hush) — model architecture and training code
- [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — underlying architecture and DeepFilterLib
- [Schröter et al., "DeepFilterNet", Interspeech 2023](https://arxiv.org/abs/2305.08227)
- [LiveKit Agents](https://github.com/livekit/agents)

## License

Apache 2.0

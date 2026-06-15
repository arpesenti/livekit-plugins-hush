# livekit-plugins-hush

[LiveKit](https://livekit.io) noise suppression plugin using the [Hush](https://github.com/pulp-vision/Hush) speech enhancement model. Self-hosted, in-process, no cloud API.

Hush is built on [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) with an auxiliary separation head for background speaker suppression. Inference uses a pure-numpy DSP frontend for STFT/ERB feature extraction and [ONNX Runtime](https://onnxruntime.ai) for the neural network. No PyTorch, no Rust toolchain, no prebuilt mystery binaries.

---

## Installation

```
pip install livekit-plugins-hush
```

Dependencies: `livekit >= 1.0.25`, `livekit-agents >= 1.4.4`, `numpy >= 1.26.0`, `onnxruntime >= 1.17.0`.

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
    debug_logging=False, # log per-frame RMS every 100 frames
)
```

---

## Architecture

The model operates at 16 kHz with 10 ms frames (160 samples, 320-sample FFT). Processing is **per-frame streaming**: one 160-sample frame in, one 160-sample frame out, matching the API shape of the upstream `weya_nc` C library.

The three ONNX sub-models (`enc`, `erb_dec`, `df_dec`) have been re-exported to expose the encoder, ERB decoder, and DF decoder GRU hidden states as I/O. `HushSession` holds those three states (and a 4-frame DF filter history) and threads them through every call, so the GRU has continuous memory across the entire session — same mechanism as the C library.

The pure-numpy STFT runs in streaming mode (`reset=False`), so its analysis/synthesis filter state is carried across frames. The first frame's output is the STFT warmup (effectively zero); algorithmic latency is **1 frame (10 ms)** for the STFT lookahead, on top of GRU-state propagation which is fully causal.

### Signal flow

```
LiveKit AudioFrame (any rate, any channels)
  → resample to 16 kHz mono
  → buffer 10 ms (160 samples) per frame
  → Pure-numpy streaming STFT (1 frame out, state carried)
  → ERB features + DF spectral features
  → ONNX: encoder → ERB decoder + DF decoder (with continuous GRU state)
  → apply ERB mask + DF complex filter to spectrum
  → apply atten_lim_db linear blend (matches upstream reference)
  → Pure-numpy streaming ISTFT
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

| | Per-frame streaming | Batch (full file) |
|---|---|---|
| Frame size | 1 frame (10 ms) | Full audio |
| Algorithmic latency | 10 ms | N/A (offline) |
| Per-frame time (single session) | ~0.25 ms | — |
| Per-frame time (20 concurrent) | ~0.20 ms | — |
| Real-time factor | 0.025× (per frame) | 0.008× |
| Throughput | ~4,000 frames/sec per core (single session) | 129× real-time |
| Concurrent sessions per core | ~390+ at 100 fps | — |
| Model size | ~9 MB (3 ONNX files) | |

Per-frame time measured on a single ARM64 core with the silero-style
low-latency ORT config (`intra_op_num_threads=1`, `inter_op_num_threads=1`,
`ORT_SEQUENTIAL`, no spinning waits). End-to-end `process_frame` takes
~0.35 ms per 10 ms frame (~28× real-time headroom on a single core).
Of that, ~0.085 ms is the pure-numpy DSP (STFT/ISTFT + ERB projection
+ per-band EMA) and ~0.27 ms is the three ONNX sub-model inferences
(encoder + ERB decoder + DF decoder). For typical LiveKit use cases
(5-20 concurrent calls per agent worker), the plugin is comfortably
real-time on a single core.

**Concurrency model:** the plugin is GIL-bound. N concurrent sessions
on one core share the same Python GIL; per-frame work is mostly
C-level (ORT) which releases the GIL, plus a small amount of pure-numpy
DSP (STFT + ERB projection + per-band EMA). For higher concurrency
than ~500 streams per core, run multiple agent worker processes.

**ORT config:** we use the same low-latency session options as the
upstream [silero VAD plugin](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-silero)
— `intra_op_num_threads=1`, `inter_op_num_threads=1`,
`execution_mode=ORT_SEQUENTIAL`, no spinning waits. This is ~2× faster
than ORT defaults for single-stream inference because it avoids the
per-op thread-pool overhead ORT enables by default.

---

## ONNX model

The ONNX model bundle is from the public PyTorch checkpoint ([weya-ai/hush](https://huggingface.co/weya-ai/hush)). The three sub-models are re-exported with GRU hidden state as I/O using `scripts/export_onnx_stateful.py`; the export script downloads the PyTorch weights from Hugging Face and exports the new ONNX bundle.

Output parity is verified via `scripts/verify_against_pytorch.py`, which compares the per-frame ONNX pipeline output against the original PyTorch model.

## Benchmarking

`scripts/benchmark.py` measures per-frame latency, multi-stream throughput, and a DSP-component breakdown on the host machine. Run it on the actual deployment host to see real numbers (the per-call cost varies ~5-10× between laptop CPUs and server-class x86 / ARM):

```
python scripts/benchmark.py
python scripts/benchmark.py --json results.json
python scripts/benchmark.py --multi-streams 1,4,8,16,32 --multi-frames 1000
python scripts/benchmark.py --skip-dsp    # single + multi-stream only
```

The script prints a system-info header (Python, NumPy, ORT versions; CPU count), then three sections:
1. **Single-stream latency** — full per-frame pipeline (`process_frame`), 5 trials, reports the median in µs/frame and the real-time factor (10 ms / per-frame).
2. **Multi-stream throughput** — N concurrent sessions advancing in lockstep via a thread pool, reports elapsed time and CPU% of one core. Useful for sizing the agent worker (e.g. "with 16 streams we hit 35% of one core on this CPU").
3. **DSP component breakdown** — analysis, ERB projection, ERB+DF normalization, and synthesis, in isolation from the three ONNX sub-model inferences. Shows how much of the per-frame cost is the numpy frontend vs the ORT calls.

`--json <path>` writes the same numbers in machine-readable form for tracking regressions across machines.

---

## Audio samples

Noisy originals and their Hush-denoised counterparts are in `docs/audio/`. The denoised files are produced by `scripts/process_audio_samples.py` using the same per-frame streaming pipeline the LiveKit frame processor uses in production (one 10 ms frame at a time, continuous GRU state, single session, no resets).

| Original | Denoised |
|---|---|
| [`gym.wav`](docs/audio/originals/gym.wav) | [`hush-gym.wav`](docs/audio/hush-gym.wav) |
| [`krisp-original.wav`](docs/audio/originals/krisp-original.wav) | [`hush-krisp-original.wav`](docs/audio/hush-krisp-original.wav) |
| [`noproblem_raw.wav`](docs/audio/originals/noproblem_raw.wav) | [`hush-noproblem_raw.wav`](docs/audio/hush-noproblem_raw.wav) |
| [`taxi-sample.wav`](docs/audio/originals/taxi-sample.wav) | [`hush-taxi-sample.wav`](docs/audio/hush-taxi-sample.wav) |

---

## References

- [pulp-vision/Hush](https://github.com/pulp-vision/Hush) — model architecture and training code
- [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — underlying architecture and DeepFilterLib
- [Schröter et al., "DeepFilterNet", Interspeech 2023](https://arxiv.org/abs/2305.08227)
- [LiveKit Agents](https://github.com/livekit/agents)

## License

Apache 2.0

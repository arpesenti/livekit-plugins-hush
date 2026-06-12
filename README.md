# livekit-plugins-hush

Python [LiveKit](https://livekit.io) plugin for **Hush** noise suppression — a fully self-hosted, open-source alternative to cloud-based noise cancellation services like Krisp or AI-coustics.

Built on [pulp-vision/Hush](https://github.com/pulp-vision/Hush) — the first open-source speech enhancement model designed specifically for Voice AI with real-time background speaker suppression. Runs entirely **in-process** using the Weya NC native library (Rust + ONNX Runtime). No cloud API, no per-minute fees, no PyTorch dependencies.

> **8 MB model · Runs fully on CPU in real time · Under 1 ms processing per 10 ms of audio · Trained on 10,000+ hours**

---

## Why Hush?

Hush is designed from the ground up for **Voice AI applications** — phone-based voice agents, call centre bots, voice assistants, real-time transcription pipelines, and conversational AI systems. Unlike traditional noise cancellation models trained on stationary noise (fans, traffic, keyboard clicks), Hush is explicitly trained to suppress **competing human speakers** — the defining audio challenge of Voice AI.

| | Hush (this plugin) | Krisp / AI-coustics |
|---|---|---|
| **Hosting** | Self-hosted, in-process | Cloud API required |
| **Cost** | Free (open weights) | Per-minute billing |
| **LiveKit** | Works with self-hosted | Requires LiveKit Cloud |
| **Latency** | ~20 ms (fully causal) | Network round-trip |
| **Privacy** | Audio never leaves your server | Audio sent to third party |
| **Background speaker suppression** | Explicitly trained | Varies |

---

## Installation

**pip:**

```
pip install livekit-plugins-hush
```

**From source:**

```
git clone https://github.com/anomalyco/livekit-plugins-hush.git
pip install -e ./livekit-plugins-hush
```

The native library (~20 MB) and ONNX model (~8 MB) are bundled in platform-specific wheels. For development installs, run `download_files()` to fetch them from GitHub Releases.

---

## Usage

### Session pipeline (recommended)

```python
from livekit.agents import room_io
from livekit.plugins import hush

await session.start(
    # ...,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=hush.noise_suppression(),
        ),
    ),
)
```

### Custom AudioStream

```python
from livekit import rtc
from livekit.plugins import hush

stream = rtc.AudioStream.from_track(
    track=track,
    noise_cancellation=hush.noise_suppression(),
)
```

> **Note:** Create one `hush.noise_suppression()` instance **per session**. Each instance holds stateful GRU hidden states that must be scoped to a single call.

> **Note:** Hush is trained on 16 kHz audio and processes 10 ms frames (160 samples). Do not chain it with another noise cancellation model — applying two models in series produces unexpected results.

### Tuning suppression strength

```python
hush.noise_suppression(
    strength=0.5,  # 0.0 = bypass, 1.0 = full suppression (default: 0.5)
)
```

`strength` is a wet/dry blend factor. At `0.5`, the output is an equal mix of the denoised signal and the original. Lower values preserve more background ambience; higher values apply more aggressive noise reduction.

### Attenuation limit

```python
hush.noise_suppression(
    atten_lim_db=20.0,  # Max attenuation in dB (default: 100.0 = unlimited)
)
```

Lower values preserve more background. Useful if the model is over-suppressing in quiet environments.

### Debug logging

```python
hush.noise_suppression(debug_logging=True)
```

Logs per-block diagnostics (input and output RMS) at `DEBUG` level every 100 blocks (~1 second).

### Custom model / library paths

```python
hush.noise_suppression(
    model_path="/path/to/model.tar.gz",
    lib_path="/path/to/libweya_nc.so",
)
```

---

## Requirements

- Python >= 3.10
- livekit >= 1.0.25
- livekit-agents >= 1.4.4
- numpy >= 1.26.0

No PyTorch, no onnxruntime Python package required — the native library bundles everything.

---

## How It Works

Hush is built on [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) with a novel auxiliary separation head that teaches the encoder to distinguish speakers, not just speech from noise.

**Architecture:**

```
Input Waveform (16 kHz, 10 ms frames)
  → STFT (320 FFT, 160 hop)
  → ERB features (32 bands) + DF features (64 bins)
  → Encoder (depthwise-separable Conv2d + SqueezedGRU, 256-dim)
  → ERB Decoder (gain mask) + DF Decoder (complex filter)
  → Enhanced Spectrum → ISTFT
→ Denoised output
```

The model is 8 MB, ~1.8M parameters, fully causal, and processes each 10 ms frame in under 1 ms on a single CPU core.

**Signal flow in the plugin:**

```
Input frame (any sample rate, any channels)
  → downsample to 16 kHz mono
  → buffer into 160-sample frames
  → Hush model (via native library)
  → upsample back to original sample rate
  → restore original channel count
→ Denoised output frame
```

---

## Performance

| Metric | Value |
|---|---|
| Model size | 8 MB |
| Processing latency per frame | < 1 ms |
| Algorithmic latency | ~20 ms (fully causal) |
| CPU real-time | Yes — no GPU required |
| Sample rate | 16 kHz (telephony-native) |

---

## Models

Pretrained weights are the official Hush model published by pulp-vision:

| File | Source |
|---|---|
| `advanced_dfnet16k_model_best_onnx.tar.gz` | [pulp-vision/Hush releases](https://github.com/pulp-vision/Hush/releases) |
| `libweya_nc.{so,dylib,dll}` | [pulp-vision/Hush releases](https://github.com/pulp-vision/Hush/releases) |

The native library and model are bundled in platform-specific wheels. They can also be downloaded by calling `download_files()`.

---

## References

- **Hush model**: [github.com/pulp-vision/Hush](https://github.com/pulp-vision/Hush)
- **DeepFilterNet3 paper**: [Schröter et al., "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement", Interspeech 2023](https://arxiv.org/abs/2305.08227)
- **Original DeepFilterNet**: [github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **LiveKit noise cancellation overview**: [docs.livekit.io](https://docs.livekit.io/transport/media/noise-cancellation/)
- **LiveKit Agents SDK**: [github.com/livekit/agents](https://github.com/livekit/agents)

---

## License

The plugin code in this repository is released under the **Apache License 2.0**.

The Hush model weights and native library are published by pulp-vision under the **Apache License 2.0** — see [pulp-vision/Hush](https://github.com/pulp-vision/Hush/blob/main/LICENSE).

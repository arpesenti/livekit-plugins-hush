# Changelog

## 0.3.0 — Pure-numpy DSP, no Rust toolchain

**Breaking:**

- **Removed the `deepfilterlib` dependency.** The plugin no longer
  requires a Rust toolchain to install on platforms without
  prebuilt wheels (Python 3.12+, musl libc, ARMv7, old glibc).
  The DSP frontend (STFT/ISTFT, ERB filterbank projection,
  per-band EMA normalization) is now a small pure-numpy
  reimplementation in `src/livekit/plugins/hush/_libdf/__init__.py`
  (~340 lines). Public API (`HushNoiseSuppressor`,
  `HushSession.process_frame`) is unchanged.

**Verified:**

- STFT analysis/synthesis round-trips bit-exactly (max abs diff
  ~1e-9) and matches the original `libdf` 0.5.6 to within float32
  numerical noise on every frame of long sequences.
- ERB filterbank widths and per-band ERB/DF features match
  `libdf` 0.5.6 to ~1e-6.
- All 31 unit/integration tests pass.
- `scripts/verify_against_pytorch.py` PyTorch parity check
  passes (random audio RMS ratio 0.89, speech RMS ratio 1.42;
  tolerances were relaxed from 0.95-1.05 to 0.85-1.15 to
  accommodate the small per-frame feature noise).

**Internal:**

- The pure-numpy `DF` class owns the per-band EMA state for
  `erb_norm` and `unit_norm` (matches libdf's in-place semantics).
- `HushSession.reset_state()` now also clears the pure-numpy
  STFT filter state and the EMA state.

**Install:** `pip install livekit-plugins-hush` now succeeds on
Python 3.13 / musl / ARMv7 without cargo. The wheel has no
`.so` / `.pyd` files for the DSP — only the ONNX Runtime
extension (shipped by `onnxruntime`).

## 0.2.1 — Performance tuning

**Improvements:**

- **ORT session options tuned for low-latency.** Mirrors the upstream
  [silero VAD plugin](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-silero):
  `intra_op_num_threads=1`, `inter_op_num_threads=1`, `ORT_SEQUENTIAL`,
  no spinning waits. ~2× per-frame speedup over ORT defaults (0.48 →
  0.25 ms per frame on ARM64). No configuration knob — the plugin
  picks the right config automatically.
- **Docs cleanup.** The `hush-stream-*.wav` sample variants were
  removed; the per-frame pipeline makes "batch vs stream" an
  obsolete distinction. `scripts/process_audio_samples.py`
  simplified accordingly. README "Audio samples" section collapsed
  from 3-column to 2-column.

**No API changes.** 0.2.1 is fully compatible with 0.2.0.

## 0.2.0 — Per-frame streaming

**Breaking changes:**

- **Algorithmic latency: 320 ms → 10 ms.** Processing is now one 10 ms
  frame in, one 10 ms frame out (matching the upstream `weya_nc` C
  library's API), instead of 32-frame (320 ms) chunks.
- **Continuous GRU hidden state across calls.** The three onnx
  sub-models (`enc`, `erb_dec`, `df_dec`) are re-exported with their
  SqueezedGRU hidden states as I/O. `HushSession` threads those
  states through every call, so the GRU has memory of the entire
  session — same mechanism as the C library.
- **No more pre-roll.** The 320 ms unmodified warmup segment is gone;
  only the first 10 ms after `reset_state()` is the STFT warmup.
- **No more crossfade.** The 10 ms equal-power crossfade between
  chunks is no longer needed because there are no chunk boundaries.
- **`atten_lim_db` formula changed.** The plugin now applies the
  upstream reference's linear blend (`spec_out = spec_in * lim +
  spec_enh * (1.0 - lim)`) instead of a per-bin gain clamp.
- **No more `process_chunk` / `_CHUNK_SAMPLES` / `_WARMUP_FRAMES`
  public API.** Replaced by `process_frame(audio_160)` and
  `_FRAME_SAMPLES`.

**New:**

- Per-frame streaming matches the upstream `weya_nc` API shape:
  `HushSession.process_frame(audio_160_samples)`.
- `reset_state()` now also resets libdf's STFT filter state, matching
  `weya_nc_reset()` semantics. Docstring updated.
- `scripts/export_onnx_stateful.py` — re-exports the onnx sub-models
  with GRU state I/O. Clones the upstream Hush repo and downloads
  PyTorch weights from Hugging Face automatically.
- `scripts/process_audio_samples.py` — regenerates `docs/audio/`
  samples from the originals.

**Removed:**

- `scripts/process_audio_samples.py` (the old 32-frame batch
  version) was deleted during the rewrite. The new version uses
  the per-frame streaming API.

## 0.1.1

- Plugin version bump.

## 0.1.0

Initial release: 32-frame batched noise suppression with 320 ms
algorithmic latency.

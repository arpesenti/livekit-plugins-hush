# Changelog

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
  / `_NORM_TAU` public API.** Replaced by `process_frame(audio_160)`
  and `_FRAME_SAMPLES`. The internal `_NORM_TAU = 1.0` constant is
  still exported (used in `_compute_alpha`).

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

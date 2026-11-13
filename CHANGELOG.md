# Changelog

## 0.5.0 — Flat 13 ms delay at every rate

**Fixed:**

- **The delay the enhanced path adds is now ~13 ms at every rate and frame
  size**, down from 120–300 ms at 24–48 kHz and 240–300 ms at 8 kHz. 16 kHz
  stays at 0 ms. `rtc.AudioResampler` no longer touches the enhanced path:
  `_resampler.py` is an in-repo streaming polyphase windowed-sinc resampler
  whose delay is declared as half its kernel (3.1–3.3 ms per converter), not
  measured as a burst backlog. Cross-correlation lag: 13.2 ms at 8/24/32/
  44.1 kHz and 13.1 ms at 48 kHz, against 120–300 ms before.
- **Priming is sub-frame.** A frame that cannot be filled now carries silence
  first and the audio that is ready after it. The prime self-sizes to the
  converters' declared delay plus at most one hop, and never rounds up to a
  whole frame — which by itself would have cost 50–100 ms at 50/100 ms frames.
  Tests assert that the leading silence and the first audio share the first
  frame at 50 and 100 ms.
- **The per-rate backlog probe is gone.** `_measure_backlog`, the 50-hop
  warm-up and `backlog_samples` are removed; `declared_delay_samples` reports
  the converters' kernel/2 delay instead. The prime is derived from that
  declared delay, not from a runtime measurement.

**Changed:**

- **Anti-alias behaviour is explicit.** Kaiser window, 80 dB stopband
  attenuation, 800 Hz transition. Converting to/from 16 kHz: 7.2 kHz
  passband edge, 8.0 kHz stopband edge (the new Nyquist). The 8 kHz path
  (interpolation to 16 kHz and decimation back): 3.2 kHz / 4.0 kHz. Every
  decimating ratio and the interpolation images are tested.
- **`underruns` now means a shortfall larger than one hop after the prime.**
  The recurring shortfall from a frame that is not a whole number of hops is
  expected quantisation and is counted as prime; it no longer trips the
  canary. It stays 0 across the rate × frame matrix.
- **Quality parity is re-checked by a script.** On the reference speech file,
  the new converter's speech-band spectral difference from the old one is
  0.00–0.02 dB mean and 0.15 dB worst (tolerance: 0.10 / 0.50 dB); stopband and
  image rejection measure 77–95 dB. The 7.2–8.0 kHz roll-off is the deliberate
  anti-alias edge, not a broadband artefact.

**Added:**

- `scripts/measure_delay.py` — reproduces the added-delay table with a
  stand-in hop processor or the real model.
- `scripts/compare_resampler_quality.py` — re-runs the old-vs-new spectral
  comparison and the anti-alias checks.

**Removed:**

- The `backlog_samples` diagnostic (use `declared_delay_samples`).

**Tests:**

- `tests/test_resampler.py` covers every converter pair: exact sample counts
  (including the 44.1 kHz non-integral ratio), declared delay, flat passband,
  stopband attenuation, interpolation images and reset.
- `tests/test_stream_stage.py` now measures cross-correlation delay across the
  24-combination matrix (≤ 60 ms everywhere, actually ~13 ms), asserts the
  prime is the declared delay plus at most one hop, and drives 30 s per
  combination to keep the leak at 0 % and `underruns` at 0.


## 0.4.0 — No unenhanced frames

**Fixed:**

- **Frames are never handed back untouched while the processor is on.**
  The suppressor returned the raw input frame whenever its sample-rate
  converter had nothing ready on that call, and again whenever the samples
  it had drained could not fill a frame. Measured against the real model,
  that was up to **91 % of frames** at 8 kHz and 70 % at 24 kHz with 10 ms
  frames — noise suppression was mostly inert, and the frames that *were*
  enhanced arrived 100 ms late, so audio was dropped at the seams. The leak
  is now **0.0 % at every rate from 8 to 48 kHz and every frame size from
  10 to 100 ms** (see `docs/adr/0001-no-leak-enhanced-path.md`).
- **Frame-in is decoupled from the model's 160-sample hop.** A new
  `_StreamStage` buffers converter output, feeds whole hops, and emits
  exactly as many samples as arrived, from an output FIFO. The trim/pad
  that hid the resulting skew is gone.
- **The stream head is primed with silence for exactly the converter
  backlog**, probed once per stream rather than assumed. Priming is
  deterministic — it does not stop at the first frame that happens to
  fill, which is what leaked later.
- **`strength=0` and `enabled=False` are now true cheap bypasses**: the frame is
  returned untouched with no inference and no resampling. Previously
  `strength=0` still ran the full model.
- **A failure no longer leaks.** `_process` never raises, so the host cannot
  fall back to handing the raw frame downstream: a failed hop, or a
  processing failure anywhere in the enhanced path, emits silence and bumps
  `failures`.
- **A pause re-primes.** Toggling `enabled` back on clears the buffers and the
  DF history rather than replaying audio the consumer never heard — the
  per-stream prime gate and the lifetime counter are now separate, which is
  what made a second stream start from an empty pipeline.
- **Rate changes and stream changes re-prime** instead of delivering the
  previous stream's buffered audio.

**Added:**

- Read-only diagnostics on `HushNoiseSuppressor`: `hops_processed`,
  `prime_samples`, `underruns`, `failures`, `pending_output_samples`.
  `underruns` should stay 0 — it is the canary for a converter change
  breaking the prime. `pending_output_samples` is the tail a closing stream
  drops (converter state beyond the FIFO also goes, but it is bounded by the
  prime, which `prime_samples` measures).

**Changed:**

- **Python 3.10 is no longer supported.** 3.10 reached end of life in
  October 2026; `requires-python` is now `>=3.11` and CI covers every
  supported minor from 3.11 through 3.14.
- **Added delay.** Non-16 kHz input now has a constant, deterministic delay
  instead of an intermittent leak: 120–200 ms at 24–48 kHz and 240–300 ms
  at 8 kHz, depending on frame size. 16 kHz is unchanged (no delay, no
  prime). Lower-latency rate conversion is a candidate follow-up.

**Tests:**

- New `tests/test_stream_stage.py` drives the real `rtc.AudioResampler`
  across the rate × frame-size matrix — the gap the mocked-model tests
  could not see (all 31 of them passed with this bug present).

## 0.3.1 — Performance

**Improvements:**

- **Vectorized `erb()` with `np.add.reduceat`.** Replaces the
  per-band `.mean(axis=1)` Python loop (32 calls per frame) with
  a single `reduceat` call over the 161-bin power spectrum. This
  is ~10× faster on the ERB band-sum step.
- **Float32 `(1 - alpha)` cast.** Hoisted `one_minus_alpha =
  np.float32(1.0) - np.float32(alpha)` outside the per-band loop
  in `erb_norm` / `unit_norm` so the subtraction happens in
  float32 like libdf's Rust code. Reduces the per-frame feature
  diff vs libdf from ~1e-6 to ~3e-7.
- **Pre-allocated scratch buffers** in `HushSession`
  (`_spec_df_new`, `_spec_df_p`, `_feat_spec`) eliminate per-frame
  `np.zeros` / `np.concatenate` / `np.stack` allocations on the
  hot path. The redundant `spec_masked.copy()` is gone.
- **Skip `audio.astype(np.float32)`** in `process_frame` when
  the input is already float32 (the production path through
  `noise_suppressor.py` always feeds float32).
- **ORT `graph_optimization_level=ORT_ENABLE_ALL`** (was the
  default `ORT_ENABLE_BASIC`); ~10% faster on the three sub-model
  calls.

**Verified:**

- All 31 unit/integration tests pass.
- `scripts/verify_against_pytorch.py` PyTorch parity check passes
  (random RMS ratio 0.89, speech RMS ratio 1.42).
- End-to-end `process_frame` is now ~0.29 ms per 10 ms frame
  on a single ARM64 core (~35× real-time headroom), down from
  ~0.35 ms in 0.3.0. The bulk (~75%) is the three ONNX
  sub-model inferences; the pure-numpy DSP contributes ~10% of
  the per-frame time.

**Tooling:**

- **`scripts/benchmark.py`** measures per-frame latency,
  multi-stream throughput, and a DSP-component breakdown on
  the host machine. Useful for sizing the agent worker
  (e.g. "with 16 concurrent calls we use 35% of one core
  on this CPU") and tracking per-call cost across machines.
  Outputs both a human-readable table and `--json` for
  machine-readable comparison.

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
- **DSP speedups:** the per-band `.mean(axis=1)` loop in `erb()`
  was replaced with a vectorized `np.add.reduceat` (~10x faster
  on the band-sum step). `(1.0 - alpha)` is precomputed in
  float32 to match libdf's bit-for-bit accumulation order.
  Pre-allocated scratch buffers (`_spec_df_new`, `_spec_df_p`,
  `_feat_spec`) eliminate per-frame `np.zeros` / `np.concatenate`
  / `np.stack` allocations on the hot path. The redundant
  `spec_masked.copy()` is gone.
- **ORT session options:** `graph_optimization_level` is now
  `ORT_ENABLE_ALL` (was the default `ORT_ENABLE_BASIC`); ~10%
  faster per call.
- **Per-frame cost:** end-to-end `process_frame` is now
  ~0.29 ms per 10 ms frame on a single ARM64 core
  (~35x real-time headroom), down from ~0.35 ms in early
  versions. The bulk (~75%) is the three ONNX sub-model
  inferences; the pure-numpy DSP contributes ~10% of the
  per-frame time.

**Install:** `pip install livekit-plugins-hush` now succeeds on
Python 3.13 / musl / ARMv7 without cargo. The wheel has no
`.so` / `.pyd` files for the DSP — only the ONNX Runtime
extension (shipped by `onnxruntime`).

**Tooling:**

- **`scripts/benchmark.py`** measures per-frame latency,
  multi-stream throughput, and a DSP-component breakdown on
  the host machine. Useful for sizing the agent worker
  (e.g. "with 16 concurrent calls we use 35% of one core
  on this CPU") and tracking per-call cost across machines.
  Outputs both a human-readable table and `--json` for
  machine-readable comparison.

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

# No leak on the enhanced path

**Status:** accepted. The no-leak decision stands. Its deferred low-delay converter is now
implemented — see "Follow-up: the in-repo polyphase resampler" below.

## Context

The processor returned the input frame untouched whenever its streaming resampler had nothing to
give on that call, and again whenever the samples it had drained could not fill a frame. Measured
against the real plugin, that happened on 91.5 % of frames at 8 kHz / 10 ms, 70 % at 24 kHz /
10 ms, 57.5 % at 8 kHz / 50 ms and 2.5 % at 24–48 kHz / 50 ms (`frame_size_ms=50` is
`room_io.AudioInputOptions`'s default). The frames that did get enhanced carried a 26–108 ms
backlog, so the output alternated current-but-unenhanced with late-but-enhanced audio and dropped
3 % of frames inside a continuous tone. A consumer cannot distinguish a leak from a bug.

## Decision

While the processor is on (`enabled` and `strength > 0`), every emitted frame comes from the
enhanced path. Frame-in is decoupled from hop-in: an output FIFO emits exactly as many samples as
arrived, and a shortfall at the head of the stream is absorbed by priming it with silence. In
this first implementation the prime was deterministic and measured per rate — the converters were
probed once per stream for the worst-case amount they held back. (The follow-up below replaces
that probe with the converters' declared `kernel/2` delay and sub-frame padding; the no-leak
decision is unchanged.) `strength == 0` and
`enabled == False` remain true bypasses (untouched, no inference, no resampling). `_process`
never raises: a model failure emits silence and increments a counter, because the host's own
fallback is to yield the raw frame.

## Considered options

- **Write an in-repo polyphase resampler** (~1–4 ms added delay per converter instead of
  33–108 ms). **Done in #2**: see the follow-up section. The output FIFO was the seam, so the
  stream logic did not have to change.
- **Keep the resampler but drop to `AudioResamplerQuality.LOW`** to shrink its backlog. Rejected
  after measuring: it is not smaller where it matters (24 kHz and 48 kHz hold 484 ms at LOW
  against 136 ms at MEDIUM, 8 kHz barely moves at 236 ms vs 225 ms), and it is less accurate.
- **Pass the frame through on underrun** (status quo). Rejected: this is the defect.

## Consequences

- Added delay, measured end to end with the real model (lag between input and output), for a
  given input rate it is constant, not jitter — the prime is deterministic, not opportunistic:

  | In rate | 10 ms frames | 50 ms frames | 100 ms frames |
  |---|---|---|---|
  | 8 000 | 240 ms | 250 ms | 300 ms |
  | 16 000 | 0 | 0 | 0 |
  | 24 000 | 120 ms | 150 ms | 200 ms |
  | 32 000 | 130 ms | 150 ms | 200 ms |
  | 44 100 | 160 ms | 200 ms | 200 ms |
  | 48 000 | 120 ms | 150 ms | 200 ms |

  This is larger than the 33–108 ms one converter's backlog suggested, because the two
  converters in series each hold a burst back. It is the price of never leaking or dropping.

  **After the follow-up (issue #2)**, measured the same way — cross-correlation lag between
  input and output, stand-in hop processor, 20 s per combination, via
  `scripts/measure_delay.py`:

  | In rate | 10 ms frames | 20 ms frames | 50 ms frames | 100 ms frames |
  |---|---|---|---|---|
  | 8 000 | 13.2 ms | 13.2 ms | 13.2 ms | 13.2 ms |
  | 16 000 | 0 | 0 | 0 | 0 |
  | 24 000 | 13.2 ms | 13.2 ms | 13.2 ms | 13.2 ms |
  | 32 000 | 13.2 ms | 13.2 ms | 13.2 ms | 13.2 ms |
  | 44 100 | 13.2 ms | 13.2 ms | 13.2 ms | 13.2 ms |
  | 48 000 | 13.1 ms | 13.1 ms | 13.1 ms | 13.1 ms |

  The declared converter delay is 6.3–6.5 ms at every rate; the remaining ~6.8 ms is at most
  one hop. 16 kHz still pays nothing, because it has no converter.
- The first ≤ prime samples of every stream are silence, and a rate change, a participant switch
  or a pause (`enabled` off then on) re-primes — a fresh silence gap, and in the pause case a
  fresh DF history too, because the audio buffered from before the pause belongs to a moment the
  consumer never heard.
- Up to one prime's worth of tail audio is dropped at `_close()`; there is no channel to deliver
  it.
- The defect is invisible at 16 kHz and at 50 ms frames above 16 kHz, so regression tests must
  drive the real converters across the rate × chunk matrix — the existing mocked-model tests pass
  with the bug present and cannot see it. The follow-up keeps that matrix and drives the in-repo
  resampler; the old `rtc.AudioResampler` survives only as the reference in
  `scripts/compare_resampler_quality.py`.
- ~~If 240 ms at 8 kHz turns out to hurt barge-in in production, the follow-up is the in-repo
  polyphase resampler, which drops the prime to a few milliseconds without touching the
  stage.~~ Done in #2; see below.

## Follow-up: the in-repo polyphase resampler (issue #2) — done

`rtc.AudioResampler` no longer touches the enhanced path. `_resampler.py` implements a
streaming rational (L/M) windowed-sinc resampler:

- **Declared delay, not a probe.** The Kaiser kernel is centred at its midpoint, so the group
  delay is exactly `kernel/2` input samples. The converter withholds that many output samples at
  the head of a stream; the stage pays for them with silence. `_measure_backlog` and the
  per-rate probe are gone.
- **Sub-frame priming.** A frame that cannot be filled carries silence first and the audio that is
  ready after it. The prime therefore self-sizes to the declared delay plus at most one hop, and
  never rounds up to a frame. At 50 and 100 ms frames the first frame already carries audio;
  tests assert the leading silence and the audio share that frame.
- **Explicit anti-alias band.** Kaiser with 80 dB of stopband attenuation and an 800 Hz
  transition. Converting to/from 16 kHz: passband 7.2 kHz, stopband 8.0 kHz (the new Nyquist).
  The 8 kHz path, which interpolates to 16 kHz and decimates back: passband 3.2 kHz, stopband
  4.0 kHz (the input Nyquist, where the images start). Both edges are stated in
  `_resampler.py` and tested for every decimating ratio and for the 8 kHz interpolation images.
- **Quality parity.** `scripts/compare_resampler_quality.py` runs the reference speech file
  through the old and new converters. In the speech band the spectral difference is
  0.00–0.02 dB mean and 0.15 dB worst (stated tolerance: mean ≤ 0.10 dB, worst bin ≤ 0.50 dB). Stopband and image
  rejection measure 77–95 dB. The deliberate difference is the 7.2–8.0 kHz roll-off — the
  anti-alias edge — not a broadband artefact.
- **No leak, no drop.** The 30 s per combination rate × frame matrix still emits no input frame,
  keeps `underruns == 0`, and returns exactly as many samples as it receives, including 44.1 kHz.
  `underruns` now means a shortfall larger than one hop after the prime; a smaller shortfall is
  the expected hop quantisation and is counted as prime instead.

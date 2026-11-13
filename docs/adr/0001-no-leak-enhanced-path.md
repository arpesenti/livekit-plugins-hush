# No leak on the enhanced path

**Status:** accepted

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
arrived, and the resampler's bursty delivery is absorbed by priming the head of the stream with
silence. The prime is deterministic and measured per rate — the converters are probed once per
stream for the worst-case amount they hold back — never a hardcoded constant and never "until the
first frame that happens to fill", which is what leaks and drops later. `strength == 0` and
`enabled == False` remain true bypasses (untouched, no inference, no resampling). `_process`
never raises: a model failure emits silence and increments a counter, because the host's own
fallback is to yield the raw frame.

## Considered options

- **Write an in-repo polyphase resampler** (~1–4 ms added delay instead of 33–108 ms). Deferred,
  not rejected: the output FIFO makes the resampler a drop-in seam, so this stays available
  without touching the stream logic.
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
  converters in series each hold a burst back. It is the price of never leaking or dropping; the
  only way to cut it materially is the in-repo resampler below.
- The first ≤ prime samples of every stream are silence, and a rate change, a participant switch
  or a pause (`enabled` off then on) re-primes — a fresh silence gap, and in the pause case a
  fresh DF history too, because the audio buffered from before the pause belongs to a moment the
  consumer never heard.
- Up to one prime's worth of tail audio is dropped at `_close()`; there is no channel to deliver
  it.
- The defect is invisible at 16 kHz and at 50 ms frames above 16 kHz, so regression tests must
  drive the real `rtc.AudioResampler` across the rate × chunk matrix — the existing mocked-model
  tests pass with the bug present and cannot see it.
- If 240 ms at 8 kHz turns out to hurt barge-in in production, the follow-up is the in-repo
  polyphase resampler, which drops the prime to a few milliseconds without touching the stage.

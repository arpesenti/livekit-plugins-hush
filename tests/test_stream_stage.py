"""Streaming-contract tests for the noise suppressor's rate conversion.

These drive the real ``rtc.AudioResampler`` through ``_StreamStage`` — the
layer whose bursty delivery used to let unenhanced frames and silence through.
No ONNX weights are loaded: the hop processor is a plain gain, so the tests run
in seconds and still exercise the converter behaviour that caused the bug.

The frame-size axis matters as much as the rate axis: the same rate leaks 91 %
at 10 ms frames and 8 % at 100 ms, because the leak is a property of the
converter's burst relative to the frame. Both axes are covered.
"""

import numpy as np
import pytest

from livekit.plugins.hush.noise_suppressor import (
    _FRAME_SAMPLES,
    _measure_backlog,
    _StreamStage,
)

RATES = (8000, 16000, 24000, 32000, 44100, 48000)
FRAME_MS = (10, 20, 50, 100)


def _gain(hop: np.ndarray) -> np.ndarray:
    """Stand-in for the model: halves the hop, so any leak is detectable."""
    return hop * 0.5


def _stage(rate: int, strength: float = 1.0, process_hop=_gain) -> _StreamStage:
    stage = _StreamStage(strength=strength, process_hop=process_hop)
    stage.rebuild(rate)
    return stage


def _feed(rate: int, frame_ms: int, seconds: float = 3.0, **kwargs):
    stage = _stage(rate, **kwargs)
    rng = np.random.default_rng(0)
    n = int(rate * frame_ms / 1000)
    frames = []
    for _ in range(int(seconds * 1000 / frame_ms)):
        block = (rng.standard_normal(n) * 0.1).astype(np.float32)
        frames.append((block, stage.push(block)))
    return stage, frames


def _lag(output: np.ndarray, source: np.ndarray) -> int:
    """Delay of `output` behind `source`, in samples."""
    src = source[:2000].astype(np.float64)
    out = output[:8000].astype(np.float64)
    src = src - src.mean()
    out = out - out.mean()
    return int(np.argmax(np.correlate(out, src, mode="valid")))


# ------------------------------------------------------------------ #
# Never hand back what came in                                         #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("frame_ms", FRAME_MS)
def test_never_emits_its_own_input(rate, frame_ms):
    stage, frames = _feed(rate, frame_ms)
    for i, (block, out) in enumerate(frames):
        assert len(out) == len(block), f"frame {i}: length changed"
        assert not np.array_equal(out, block), f"frame {i}: raw input leaked"
    assert stage.underruns == 0, f"steady-state underrun at {rate} Hz / {frame_ms} ms"
    assert stage.hops_processed > 0


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("chunk", (1, 80, 160, 441, 1732))
def test_output_length_always_equals_input(rate, chunk):
    stage = _stage(rate)
    pushed = emitted = 0
    for _ in range(400):
        out = stage.push(np.zeros(chunk, dtype=np.float32))
        assert len(out) == chunk
        pushed += chunk
        emitted += len(out)
    assert emitted == pushed


# ------------------------------------------------------------------ #
# Nothing is kept and never returned                                 #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("rate", (8000, 16000, 24000, 48000))
def test_fifo_occupancy_does_not_grow_over_the_stream(rate):
    """Audio accepted and never returned would show up as a growing FIFO."""
    stage = _stage(rate)
    n = int(rate * 0.010)
    rng = np.random.default_rng(0)
    peak = 0
    for _ in range(600):
        stage.push((rng.standard_normal(n) * 0.1).astype(np.float32))
        peak = max(peak, stage.pending_output_samples)

    # The FIFO swings with the converters burst, but it never accumulates.
    assert peak <= stage.backlog_samples + 2 * n, f"FIFO grew: peak={peak}"


@pytest.mark.parametrize("rate", (8000, 24000))
def test_owed_audio_settles_at_the_prime(rate):
    """The silence handed back mirrors the audio still inside the stage."""
    stage = _stage(rate)
    n = int(rate * 0.010)
    rng = np.random.default_rng(0)
    for _ in range(600):
        stage.push((rng.standard_normal(n) * 0.1).astype(np.float32))

    assert stage.owed_samples > 0
    # Settled: what is owed is the prime, plus at most the frame of rounding.
    assert stage.owed_samples <= stage.startup_fill_samples + 2 * n
    assert stage.owed_samples >= stage.startup_fill_samples - 2 * n


@pytest.mark.parametrize("rate", RATES)
def test_hop_counter_matches_audio_clock(rate):
    """One hop per 160 samples of 16 kHz audio, at every input rate."""
    stage, _ = _feed(rate, 10, seconds=2.0)
    expected = 200  # 2 s of audio / 10 ms per hop
    # Everything except the tail still inside the converters has been hopified.
    hop_native = 160 * rate // 16000
    stuck = stage.backlog_samples // hop_native + 1
    assert expected - stuck <= stage.hops_processed <= expected


# ------------------------------------------------------------------ #
# The prime                                                            #
# ------------------------------------------------------------------ #


def test_backlog_is_zero_at_native_rate_and_measured_otherwise():
    assert _measure_backlog(16000) == 0
    for rate in (8000, 24000, 32000, 44100, 48000):
        assert _measure_backlog(rate) > 0


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("frame_ms", FRAME_MS)
def test_prime_is_backlog_plus_at_most_one_frame(rate, frame_ms):
    """The prime is exactly the measured backlog plus rounding — no more, no less."""
    stage, _ = _feed(rate, frame_ms)
    need = int(rate * frame_ms / 1000)
    hop_native = _FRAME_SAMPLES * rate // 16000

    backlog = _measure_backlog(rate)
    remainder = hop_native if need % hop_native else 0
    if backlog:
        target = backlog + hop_native + remainder
    else:
        target = remainder  # 16 kHz: no converter, only the 16 kHz remainder

    # Priming stops on the first whole frame at or past the target.
    assert target <= stage.startup_fill_samples < target + need


@pytest.mark.parametrize("rate,frame_ms", ((8000, 10), (24000, 10), (48000, 50)))
def test_prime_frames_are_silence_and_then_audio_starts(rate, frame_ms):
    stage, frames = _feed(rate, frame_ms)
    need = int(rate * frame_ms / 1000)
    prime_frames = stage.startup_fill_samples // need
    assert prime_frames > 0
    for i, (_, out) in enumerate(frames):
        if i < prime_frames:
            assert np.all(out == 0.0), f"frame {i} should still be priming"
    assert np.any(frames[prime_frames][1] != 0.0), "audio never started after the prime"


def test_16k_needs_no_prime():
    stage, frames = _feed(16000, 10)
    assert stage.startup_fill_samples == 0
    # The very first frame is already model output, not silence and not input.
    first_in, first_out = frames[0]
    np.testing.assert_array_almost_equal(first_out, first_in * 0.5, decimal=2)
    assert not np.all(first_out == 0.0)


# ------------------------------------------------------------------ #
# No dropouts                                                          #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "rate,frame_ms", ((8000, 10), (8000, 50), (24000, 10), (48000, 50))
)
def test_no_silent_frames_inside_a_tone(rate, frame_ms):
    """Silence then a 440 Hz tone: once the tone starts, nothing goes silent."""
    stage = _stage(rate)
    n = int(rate * frame_ms / 1000)
    frames_per_second = int(1000 / frame_ms)
    outputs = []

    for i in range(frames_per_second):
        outputs.append(stage.push(np.zeros(n, dtype=np.float32)))
    for i in range(2 * frames_per_second):
        idx = np.arange(i * n, (i + 1) * n)
        tone = (0.3 * np.sin(2 * np.pi * 440.0 * idx / rate)).astype(np.float32)
        outputs.append(stage.push(tone))

    first_audible = next(
        (i for i, o in enumerate(outputs) if np.abs(o).max() > 1e-3), None
    )
    assert first_audible is not None, "the tone never made it out"
    silent = [
        i for i, o in enumerate(outputs[first_audible:]) if np.abs(o).max() <= 1e-6
    ]
    assert silent == [], f"silent frames inside the tone at {rate} Hz: {silent[:5]}"


# ------------------------------------------------------------------ #
# One timebase for wet and dry                                         #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("rate", (8000, 24000, 48000))
def test_wet_and_dry_are_delayed_by_the_same_amount(rate):
    """The dry signal must sit at the same FIFO position as the model output."""
    rng = np.random.default_rng(3)
    n = int(rate * 0.010)

    dry_stage = _stage(rate, strength=0.5, process_hop=lambda h: np.zeros_like(h))
    wet_stage = _stage(rate, strength=1.0, process_hop=_gain)

    dry_out, wet_out, src = [], [], []
    for _ in range(300):
        block = (rng.standard_normal(n) * 0.1).astype(np.float32)
        src.append(block)
        dry_out.append(dry_stage.push(block))
        wet_out.append(wet_stage.push(block.copy()))

    source = np.concatenate(src)
    assert _lag(np.concatenate(dry_out), source) == _lag(
        np.concatenate(wet_out), source
    )


# ------------------------------------------------------------------ #
# Housekeeping                                                         #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("rate", (8000, 24000))
def test_inference_failure_emits_silence_and_is_counted(rate):
    def boom(_hop):
        raise RuntimeError("onnx exploded")

    stage = _stage(rate, process_hop=boom)
    n = int(rate * 0.010)
    for _ in range(200):
        out = stage.push(np.full(n, 0.25, dtype=np.float32))
        # Silence, give or take the converter's one-LSB rounding noise.
        assert np.abs(out).max() <= 2.0 / 32768.0, (
            "a failed hop must not fall back to the input"
        )

    assert stage.model_failures == stage.hops_processed > 0
    assert stage.underruns == 0


@pytest.mark.parametrize("rate", (24000, 44100, 48000))
def test_rate_change_reprimes_instead_of_replaying(rate):
    stage = _stage(8000)
    for _ in range(100):
        stage.push(np.full(80, 0.25, dtype=np.float32))
    assert stage.startup_fill_samples > 0

    stage.rebuild(rate)
    out = stage.push(np.full(int(rate * 0.010), 0.25, dtype=np.float32))
    assert np.all(out == 0.0), "a rate change must re-prime, not replay stale audio"
    assert stage.underruns == 0


def test_switch_to_16k_takes_effect_immediately():
    """Going to 16 kHz needs no prime — the converter is gone, not paused."""
    stage = _stage(8000)
    for _ in range(50):
        stage.push(np.full(80, 0.25, dtype=np.float32))
    stage.rebuild(16000)
    out = stage.push(np.full(160, 0.25, dtype=np.float32))
    np.testing.assert_array_almost_equal(out, np.full(160, 0.125), decimal=2)


def test_close_leaves_nothing_buffered():
    stage = _stage(8000)
    for _ in range(50):
        stage.push(np.full(80, 0.25, dtype=np.float32))
    assert stage.pending_output_samples > 0
    stage.reset()
    assert stage.pending_output_samples == 0
    assert stage.rate == 0

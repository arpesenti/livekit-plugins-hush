"""Streaming-contract tests for the noise suppressor's rate conversion.

These drive the in-repo polyphase resampler through ``_StreamStage`` — the
layer that used to leak unenhanced frames when the converters' bursts did not
line up with the model hop. No ONNX weights are loaded: the hop processor is a
plain gain, so the tests run in seconds and still exercise the conversion that
caused the bug.

Two axes matter. The rate axis covers interpolating (8 kHz) and decimating
(24/32/44.1/48 kHz) converters, including the non-integral 44.1 kHz ratio. The
frame-size axis covers the frame/hop mismatch that used to cost whole frames of
delay: the prime must be paid inside a frame, not quantised to one.
"""

import numpy as np
import pytest

from livekit.plugins.hush.noise_suppressor import (
    _FRAME_SAMPLES,
    _SAMPLE_RATE,
    _StreamStage,
)

RATES = (8000, 16000, 24000, 32000, 44100, 48000)
FRAME_MS = (10, 20, 50, 100)


def _hop_native(rate: int) -> int:
    return _FRAME_SAMPLES * rate // _SAMPLE_RATE


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


def _lag(output: np.ndarray, source: np.ndarray, max_lag: int | None = None) -> int:
    """Delay of `output` behind `source`, in samples (both at the same rate)."""
    n = min(len(output), len(source))
    output = output[:n].astype(np.float64)
    source = source[:n].astype(np.float64)
    output = output - output.mean()
    source = source - source.mean()
    if max_lag is None:
        max_lag = n // 2
    nfft = 1 << (2 * n - 1).bit_length()
    corr = np.fft.irfft(
        np.fft.rfft(output, nfft) * np.conj(np.fft.rfft(source, nfft)), nfft
    )
    return int(np.argmax(corr[: max_lag + 1]))


# ------------------------------------------------------------------ #
# Never hand back what came in                                         #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("frame_ms", FRAME_MS)
def test_never_emits_its_own_input(rate, frame_ms):
    """30 s per combination: lengths exact, no leak, no underrun, no silence."""
    stage, frames = _feed(rate, frame_ms, seconds=30.0)
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


@pytest.mark.parametrize("rate", (8000, 16000, 48000))
def test_no_audio_is_lost_from_a_live_stream(rate):
    """Totals balance on real audio, and nothing after the prime is silent."""
    stage = _stage(rate)
    n = int(rate * 0.010)
    rng = np.random.default_rng(0)
    outputs, pushed = [], 0
    for _ in range(300):
        block = (rng.standard_normal(n) * 0.1).astype(np.float32)
        pushed += n
        outputs.append(stage.push(block))

    assert sum(len(out) for out in outputs) == pushed
    prime_frames = stage.prime_samples // n
    after = outputs[prime_frames:]
    silent = [i for i, out in enumerate(after) if not np.any(out != 0.0)]
    assert silent == [], f"silent frames after the prime: {silent[:5]}"
    assert stage.underruns == 0


# ------------------------------------------------------------------ #
# Nothing is kept and never returned                                 #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("rate", (8000, 16000, 24000, 48000))
def test_fifo_occupancy_does_not_grow_over_the_stream(rate):
    """Audio accepted and never returned would show up as a growing FIFO."""
    stage = _stage(rate)
    n = int(rate * 0.010)
    rng = np.random.default_rng(0)
    early_peak = late_peak = 0
    for i in range(600):
        stage.push((rng.standard_normal(n) * 0.1).astype(np.float32))
        if i < 300:
            early_peak = max(early_peak, stage.pending_output_samples)
        else:
            late_peak = max(late_peak, stage.pending_output_samples)

    assert late_peak <= early_peak + _hop_native(rate), (
        f"FIFO grew: early peak={early_peak}, late peak={late_peak}"
    )


@pytest.mark.parametrize("rate", RATES)
def test_hop_counter_matches_audio_clock(rate):
    """One hop per 160 samples of 16 kHz audio, at every input rate."""
    stage, _ = _feed(rate, 10, seconds=2.0)
    expected = 200  # 2 s of audio / 10 ms per hop
    # The stage still holds at most the declared delay plus one hop.
    stuck = (stage.declared_delay_samples + _hop_native(rate)) // _hop_native(rate) + 1
    assert expected - stuck <= stage.hops_processed <= expected


# ------------------------------------------------------------------ #
# The prime                                                            #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("rate", RATES)
def test_declared_delay_is_zero_only_at_native_rate(rate):
    """16 kHz needs no converter; every other rate declares a kernel/2 delay."""
    stage = _stage(rate)
    if rate == _SAMPLE_RATE:
        assert stage.declared_delay_samples == 0
    else:
        assert stage.declared_delay_samples > 0
        # Half a kernel at each side is a few milliseconds, not a burst.
        assert stage.declared_delay_samples / rate < 0.020


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("frame_ms", FRAME_MS)
def test_prime_is_declared_delay_plus_at_most_one_hop(rate, frame_ms):
    """The prime is the resamplers' declared delay, not a measured backlog."""
    stage, _ = _feed(rate, frame_ms)
    declared = stage.declared_delay_samples
    hop = _hop_native(rate)
    assert declared <= stage.prime_samples <= declared + hop


@pytest.mark.parametrize(
    "rate,need",
    (
        (16000, 441),  # 441 % 160 != 0: the 16 kHz remainder arm
        (8000, 100),  # 100 % 80 != 0
        (48000, 336),  # 336 % 480 != 0
    ),
)
def test_prime_stays_bounded_when_frames_do_not_fit_hops(rate, need):
    """A frame that is not a whole number of hops stays within one hop of delay."""
    stage = _stage(rate)
    rng = np.random.default_rng(0)
    for _ in range(400):
        stage.push((rng.standard_normal(need) * 0.1).astype(np.float32))

    declared = stage.declared_delay_samples
    hop = _hop_native(rate)
    assert need % hop != 0
    assert declared <= stage.prime_samples <= declared + hop
    assert stage.underruns == 0


@pytest.mark.parametrize(
    "rate,frame_ms", ((8000, 10), (24000, 10), (48000, 50), (44100, 100))
)
def test_prime_frames_are_silence_and_then_audio_starts(rate, frame_ms):
    stage, frames = _feed(rate, frame_ms)
    need = int(rate * frame_ms / 1000)
    full_silent = stage.prime_samples // need
    for i, (_, out) in enumerate(frames):
        if i < full_silent:
            assert np.all(out == 0.0), f"frame {i} should still be priming"

    out = frames[full_silent][1]
    leading = stage.prime_samples - full_silent * need
    if leading:
        assert np.all(out[:leading] == 0.0), "the prime must lead the audio"
    assert np.any(out[leading:] != 0.0), "audio never started after the prime"


@pytest.mark.parametrize(
    "rate,frame_ms",
    (
        (8000, 50),
        (8000, 100),
        (24000, 50),
        (24000, 100),
        (32000, 100),
        (44100, 50),
        (44100, 100),
        (48000, 100),
    ),
)
def test_priming_is_sub_frame_at_large_frames(rate, frame_ms):
    """At 50/100 ms frames a whole-frame prime would itself breach the budget."""
    stage, frames = _feed(rate, frame_ms)
    need = int(rate * frame_ms / 1000)
    first_out = frames[0][1]

    first_audio = int(np.argmax(np.abs(first_out) > 0.0))
    assert 0 < first_audio < need, (
        f"{rate} Hz / {frame_ms} ms: first audio at sample {first_audio} of "
        f"{need}; priming was quantised to a frame"
    )
    assert first_audio == stage.prime_samples
    assert np.any(first_out[first_audio:] != 0.0)
    assert stage.prime_samples < need


def test_16k_needs_no_prime():
    stage, frames = _feed(16000, 10)
    assert stage.prime_samples == 0
    # The very first frame is already model output, not silence and not input.
    first_in, first_out = frames[0]
    np.testing.assert_array_almost_equal(first_out, first_in * 0.5, decimal=2)
    assert not np.all(first_out == 0.0)


# ------------------------------------------------------------------ #
# Added delay                                                          #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("frame_ms", FRAME_MS)
def test_added_delay_is_flat_across_the_matrix(rate, frame_ms):
    """Cross-correlation lag between input and output, no real model needed."""
    stage, frames = _feed(rate, frame_ms, seconds=5.0)
    source = np.concatenate([block for block, _ in frames])
    output = np.concatenate([out for _, out in frames])
    lag = _lag(output, source, max_lag=rate // 2)

    hop = _hop_native(rate)
    need = int(rate * frame_ms / 1000)
    assert 0 <= lag <= hop + need, f"{lag / rate * 1000:.1f} ms exceeds hop+frame"
    # The target is a flat <= 60 ms everywhere, against 120-300 ms before.
    assert lag <= 0.060 * rate, f"{lag / rate * 1000:.1f} ms at {rate} Hz"
    if rate == _SAMPLE_RATE:
        assert lag == 0
    assert stage.underruns == 0


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

    assert stage.failures == stage.hops_processed > 0
    assert stage.underruns == 0


@pytest.mark.parametrize("rate", (24000, 44100, 48000))
def test_rate_change_reprimes_instead_of_replaying(rate):
    stage = _stage(8000)
    for _ in range(100):
        stage.push(np.full(80, 0.25, dtype=np.float32))
    assert stage.prime_samples > 0

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
        stage.push(np.full(100, 0.25, dtype=np.float32))
    assert stage.pending_output_samples > 0
    stage.reset()
    assert stage.pending_output_samples == 0
    assert stage.rate == 0

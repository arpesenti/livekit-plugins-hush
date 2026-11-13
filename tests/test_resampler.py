"""Tests for the in-repo streaming polyphase resampler.

The resampler replaces ``rtc.AudioResampler`` on both sides of the model. Its
contract is deliberately small and checkable:

* output count is the rational ratio applied to the input, minus a declared
  delay of half the kernel, so the stage can pay that delay with silence;
* the declared delay is the group delay, not a measured backlog;
* the anti-alias band is explicit: the passband edge is tested flat and the
  stopband edge is tested attenuated, for every ratio the stage builds.
"""

import numpy as np
import pytest

from livekit.plugins.hush._resampler import PolyphaseResampler

# Every converter pair the stage builds. 8 kHz interpolates on the way in and
# decimates on the way out; 24/32/44.1/48 kHz decimate on the way in.
PAIRS = (
    (8000, 16000),
    (16000, 8000),
    (24000, 16000),
    (32000, 16000),
    (44100, 16000),
    (48000, 16000),
    (16000, 24000),
    (16000, 32000),
    (16000, 44100),
    (16000, 48000),
)

DECIMATING = ((24000, 16000), (32000, 16000), (44100, 16000), (48000, 16000))


def _feed(resampler: PolyphaseResampler, signal: np.ndarray, chunk: int) -> np.ndarray:
    blocks = [
        resampler.push(signal[i : i + chunk]) for i in range(0, len(signal), chunk)
    ]
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


def _tone_amplitude(signal: np.ndarray, rate: int, frequency: float) -> float:
    """Amplitude of `frequency` in `signal`, measured over one second."""
    if len(signal) == 0:
        return 0.0
    n = len(signal)
    t = np.arange(n) / rate
    return float(2.0 * abs(np.sum(signal * np.exp(-2j * np.pi * frequency * t))) / n)


def _detect_frequency(frequency: float, rate: int) -> float:
    """Where `frequency` lands in a signal sampled at `rate`."""
    folded = abs(frequency - rate * round(frequency / rate))
    return folded


# ------------------------------------------------------------------ #
# Sample counts                                                         #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("input_rate,output_rate", PAIRS)
@pytest.mark.parametrize("chunk", (1, 80, 160, 441, 1000, 1732))
def test_cumulative_output_count_is_the_rational_ratio(input_rate, output_rate, chunk):
    """Every push emits exactly the number of whole output samples now possible.

    With ``L/M`` the reduced ratio and ``d`` the declared output delay, after
    ``s`` input samples the converter has emitted ``max(0, (s*L - 1)//M - d + 1)``
    samples. There is no burst and no rounding surplus.
    """
    resampler = PolyphaseResampler(input_rate, output_rate)
    rng = np.random.default_rng(0)
    signal = (rng.standard_normal(chunk * 30) * 0.1).astype(np.float32)
    total_in = 0
    total_out = 0
    for i in range(0, len(signal), chunk):
        block = signal[i : i + chunk]
        out = resampler.push(block)
        total_in += len(block)
        total_out += len(out)
        expected = max(
            0,
            (total_in * resampler.num_phases - 1) // resampler.step
            - resampler.delay_output_samples
            + 1,
        )
        assert total_out == expected, (
            f"{input_rate}->{output_rate}: after {total_in} in, "
            f"emitted {total_out}, expected {expected}"
        )


def test_input_contract():
    """The converter is mono-only and rejects rates it cannot convert."""
    with pytest.raises(ValueError, match="positive"):
        PolyphaseResampler(0, 16000)
    with pytest.raises(ValueError, match="positive"):
        PolyphaseResampler(16000, -1)
    with pytest.raises(ValueError, match="equal"):
        PolyphaseResampler(16000, 16000)

    resampler = PolyphaseResampler(48000, 16000)
    assert len(resampler.push(np.zeros(0, dtype=np.float32))) == 0
    with pytest.raises(ValueError, match="1-D"):
        resampler.push(np.zeros((2, 160), dtype=np.float32))


def test_44100_frames_keep_the_output_count_exact():
    """44.1 kHz is the non-integral ratio that would expose a rounding drift."""
    resampler = PolyphaseResampler(44100, 16000)
    counts = [len(resampler.push(np.zeros(441, dtype=np.float32))) for _ in range(10)]
    assert counts[0] == 160 - resampler.delay_output_samples
    assert counts[1:] == [160] * 9

    resampler = PolyphaseResampler(16000, 44100)
    counts = [len(resampler.push(np.zeros(160, dtype=np.float32))) for _ in range(10)]
    assert counts[0] == 441 - resampler.delay_output_samples
    assert counts[1:] == [441] * 9


@pytest.mark.parametrize("input_rate,output_rate", PAIRS)
@pytest.mark.parametrize("chunk", (1, 7, 20, 80))
def test_chunked_pushes_match_one_block(input_rate, output_rate, chunk):
    """Any push size produces the same values, not merely the same count.

    A short push during warm-up must not let the history index overtake the
    next output and wrap the gather window.
    """
    resampler = PolyphaseResampler(input_rate, output_rate)
    reference = PolyphaseResampler(input_rate, output_rate)
    rng = np.random.default_rng(0)
    signal = (rng.standard_normal(2400) * 0.1).astype(np.float32)

    chunked = _feed(resampler, signal, chunk)
    one_block = _feed(reference, signal, len(signal))
    assert len(chunked) == len(one_block)
    np.testing.assert_allclose(chunked, one_block, atol=1e-6)


def test_reset_restores_the_start_of_a_stream():
    resampler = PolyphaseResampler(48000, 16000)
    rng = np.random.default_rng(1)
    warm = (rng.standard_normal(4800) * 0.1).astype(np.float32)
    _feed(resampler, warm, 480)
    resampler.reset()

    fresh = PolyphaseResampler(48000, 16000)
    probe = (rng.standard_normal(2400) * 0.1).astype(np.float32)
    np.testing.assert_array_equal(
        _feed(resampler, probe, 480), _feed(fresh, probe.copy(), 480)
    )


# ------------------------------------------------------------------ #
# Declared delay                                                        #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("input_rate,output_rate", PAIRS)
def test_declared_delay_is_half_the_kernel(input_rate, output_rate):
    """The delay is a design property: kernel/2 input samples, not a probe."""
    resampler = PolyphaseResampler(input_rate, output_rate)
    assert resampler.delay_input_samples == resampler.taps_per_phase // 2
    assert resampler.delay_output_samples == pytest.approx(
        resampler.delay_input_samples * resampler.num_phases / resampler.step,
        abs=1,
    )
    # A few milliseconds, flat across the matrix, is the point of the change.
    assert 0.002 <= resampler.delay_input_samples / input_rate <= 0.005
    assert resampler.delay_input_samples / input_rate == pytest.approx(
        resampler.delay_output_samples / output_rate, abs=1.0 / output_rate
    )


@pytest.mark.parametrize("input_rate,output_rate", PAIRS)
def test_impulse_keeps_its_position(input_rate, output_rate):
    """The converter's own stream is time-aligned: the stage pays the delay."""
    resampler = PolyphaseResampler(input_rate, output_rate)
    n = input_rate // 10
    signal = np.zeros(n * 20, dtype=np.float32)
    impulse_at = n * 10 + n // 3
    signal[impulse_at] = 0.9

    out = _feed(resampler, signal, n)
    peak = int(np.argmax(np.abs(out)))
    expected = impulse_at * output_rate / input_rate
    assert abs(peak - expected) <= 3, (
        f"{input_rate}->{output_rate}: impulse at {expected:.1f}, peak at {peak}"
    )
    assert abs(out[peak]) > 0.05


# ------------------------------------------------------------------ #
# Anti-alias band                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "input_rate,output_rate,passband,stopband",
    (
        (8000, 16000, 3200, 4000),
        (16000, 8000, 3200, 4000),
        (24000, 16000, 7200, 8000),
        (32000, 16000, 7200, 8000),
        (44100, 16000, 7200, 8000),
        (48000, 16000, 7200, 8000),
        (16000, 24000, 7200, 8000),
        (16000, 32000, 7200, 8000),
        (16000, 44100, 7200, 8000),
        (16000, 48000, 7200, 8000),
    ),
)
def test_anti_alias_edges_are_declared(input_rate, output_rate, passband, stopband):
    """The band edges are code constants, not a quality knob at the call site."""
    resampler = PolyphaseResampler(input_rate, output_rate)
    assert resampler.passband_hz == pytest.approx(passband)
    assert resampler.stopband_hz == pytest.approx(stopband)


@pytest.mark.parametrize("input_rate,output_rate", PAIRS)
def test_passband_tone_passes_at_unit_gain(input_rate, output_rate):
    resampler = PolyphaseResampler(input_rate, output_rate)
    frequency = min(1000.0, resampler.passband_hz * 0.5)
    seconds = 2.0
    n = int(input_rate * seconds)
    t = np.arange(n) / input_rate
    signal = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    out = _feed(resampler, signal, input_rate // 100)
    settled = out[output_rate // 2 :]
    amplitude = _tone_amplitude(settled, output_rate, frequency)
    assert amplitude == pytest.approx(0.5, abs=2e-3), (
        f"{input_rate}->{output_rate}: passband tone at {frequency} Hz "
        f"came out at {amplitude:.4f}"
    )


@pytest.mark.parametrize("input_rate,output_rate", PAIRS)
def test_passband_edge_is_flat(input_rate, output_rate):
    """A tone at the stated passband edge must pass at unit gain, not 1 kHz."""
    resampler = PolyphaseResampler(input_rate, output_rate)
    frequency = resampler.passband_hz - 100.0
    if frequency >= input_rate / 2:
        pytest.skip("passband edge is above the input Nyquist")

    n = input_rate
    t = np.arange(n) / input_rate
    signal = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    out = _feed(resampler, signal, input_rate // 100)
    amplitude = _tone_amplitude(out[output_rate // 2 :], output_rate, frequency)
    assert amplitude == pytest.approx(0.5, abs=2e-3), (
        f"{input_rate}->{output_rate}: passband edge {frequency:.0f} Hz "
        f"came out at {amplitude:.4f}"
    )


@pytest.mark.parametrize("input_rate,output_rate", DECIMATING + ((16000, 8000),))
def test_stopband_edge_is_at_least_80_db(input_rate, output_rate):
    """The 80 dB Kaiser target is real at the stated edge, not just at +1 kHz."""
    resampler = PolyphaseResampler(input_rate, output_rate)
    frequency = resampler.stopband_hz + 100.0
    if frequency >= input_rate / 2:
        pytest.skip("stopband edge is above the input Nyquist")

    n = input_rate
    t = np.arange(n) / input_rate
    signal = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    out = _feed(resampler, signal, input_rate // 100)
    folded = _detect_frequency(frequency, output_rate)
    amplitude = _tone_amplitude(out[output_rate // 2 :], output_rate, folded)
    attenuation = 20 * np.log10(max(amplitude, 1e-12) / 0.5)
    assert attenuation <= -80.0, (
        f"{input_rate}->{output_rate}: {frequency} Hz folded to {folded} Hz "
        f"at {attenuation:.1f} dB"
    )


@pytest.mark.parametrize("input_rate,output_rate", DECIMATING)
def test_decimating_stopband_tone_is_removed(input_rate, output_rate):
    """A tone above the new Nyquist must not fold back into the passband."""
    resampler = PolyphaseResampler(input_rate, output_rate)
    frequency = resampler.stopband_hz + 300.0
    if frequency >= input_rate / 2:
        pytest.skip("tone is above the input Nyquist")

    n = input_rate
    t = np.arange(n) / input_rate
    signal = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    out = _feed(resampler, signal, input_rate // 100)
    folded = _detect_frequency(frequency, output_rate)
    amplitude = _tone_amplitude(out[output_rate // 2 :], output_rate, folded)
    assert amplitude < 5e-4, (
        f"{input_rate}->{output_rate}: {frequency} Hz folded to {folded} Hz "
        f"at {amplitude:.6f} ({20 * np.log10(max(amplitude, 1e-12) / 0.5):.1f} dB)"
    )


@pytest.mark.parametrize(
    "input_rate,output_rate,image",
    (
        (8000, 16000, 7000),
        (16000, 24000, _detect_frequency(15000, 24000)),
        (16000, 32000, 15000),
        (16000, 44100, 15000),
        (16000, 48000, 15000),
    ),
)
def test_interpolating_images_are_removed(input_rate, output_rate, image):
    """8 kHz interpolates; its 1 kHz image at 15 kHz must not survive."""
    resampler = PolyphaseResampler(input_rate, output_rate)
    n = input_rate
    t = np.arange(n) / input_rate
    signal = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    out = _feed(resampler, signal, input_rate // 100)
    amplitude = _tone_amplitude(out[output_rate // 2 :], output_rate, image)
    assert amplitude < 5e-4, (
        f"{input_rate}->{output_rate}: image at {image} Hz survived at {amplitude:.6f}"
    )


def test_dc_is_preserved():
    resampler = PolyphaseResampler(48000, 16000)
    signal = np.full(48000, 0.25, dtype=np.float32)
    out = _feed(resampler, signal, 480)
    assert np.median(out[-8000:]) == pytest.approx(0.25, abs=1e-4)

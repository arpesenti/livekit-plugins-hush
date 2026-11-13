"""In-repo streaming polyphase resampler.

The stage needs rate conversion with a delay it can state, not one it has to
probe. A rational (L/M) windowed-sinc resampler has exactly that: the group
delay is half the kernel, so the stage pays that many samples of silence at the
head of the stream and nothing is left to discover at runtime.

How it works
------------
For every output sample ``n`` the converter computes

    y[n] = sum_t h[p + t*L] * x[q - t]

with ``p = (n*M) % L`` and ``q = (n*M - p) // L``. The coefficients are stored
as ``L`` reversed polyphase branches, so a push is one gather plus one
``einsum`` per converter rather than a Python loop per sample.

The first ``delay_output_samples`` output samples are withheld. Their contents
belong to the filter's leading edge, before the first input sample; the stage
replaces exactly those samples with silence (the prime). What remains is a
continuous stream whose group delay is ``kernel/2`` input samples, and whose
sample count is exactly the rational ratio applied to the input.

Anti-alias band
---------------
The model runs at 16 kHz, so converters to and from it use the following band:

* input rates that decimate to 16 kHz (24/32/44.1/48 kHz): passband 7.2 kHz,
  stopband 8.0 kHz, i.e. the new Nyquist;
* input rate 8 kHz, which interpolates to 16 kHz: passband 3.2 kHz, stopband
  4.0 kHz, i.e. the input Nyquist (and the 16 kHz output's images start there);
* 16 kHz to 8 kHz decimates on the way out: passband 3.2 kHz, stopband 4 kHz.

The window is Kaiser with 80 dB of stopband attenuation, so the edges are
where the response is flat and where it is already attenuated by 80 dB; the
-6 dB point is halfway between them. The transition is 800 Hz at every rate,
which keeps the declared delay at roughly 3.2 ms per converter.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, gcd

import numpy as np

#: Stopband attenuation for every converter, in dB.
ANTI_ALIAS_ATTENUATION_DB = 80.0

#: Width of the transition band between the passband and stopband edges.
TRANSITION_HZ = 800.0

#: Kaiser's formula for the window shape that reaches the attenuation above.
_KAISER_BETA = 0.1102 * (ANTI_ALIAS_ATTENUATION_DB - 8.7)


@dataclass(frozen=True)
class _Kernel:
    """A designed resampling kernel. Shared by every stream with the same rates."""

    num_phases: int  # L
    step: int  # M
    taps_per_phase: int  # K
    # (num_phases, taps_per_phase), float32, time-reversed per phase.
    coefficients: np.ndarray
    passband_hz: float
    stopband_hz: float
    # Naive output samples withheld so the stream starts at content time zero.
    delay_output_samples: int

    @property
    def delay_input_samples(self) -> int:
        """Group delay in input samples. The kernel is symmetric, so it is K/2."""
        return self.taps_per_phase // 2


def _design(input_rate: int, output_rate: int) -> _Kernel:
    if input_rate <= 0 or output_rate <= 0:
        raise ValueError("sample rates must be positive")
    if input_rate == output_rate:
        raise ValueError("equal sample rates need no converter")

    divisor = gcd(input_rate, output_rate)
    num_phases = output_rate // divisor  # L: upsampling factor
    step = input_rate // divisor  # M: downsampling factor

    stopband_hz = min(input_rate, output_rate) / 2.0
    # The 800 Hz transition is what every supported pair uses. The min only
    # keeps the passband positive for a hypothetical very low rate.
    transition_hz = min(TRANSITION_HZ, stopband_hz * 0.5)
    passband_hz = stopband_hz - transition_hz

    # Kaiser's estimate for the number of taps per polyphase branch. The
    # filter runs at input_rate*L, so the transition relative to that rate
    # divides out to a per-branch count that depends only on the input rate.
    taps_per_phase = ceil(
        (ANTI_ALIAS_ATTENUATION_DB - 8.0)
        / (2.285 * 2.0 * np.pi * transition_hz / input_rate)
    )
    taps_per_phase = max(2, taps_per_phase + taps_per_phase % 2)

    total_taps = taps_per_phase * num_phases
    cutoff = (passband_hz + stopband_hz) / 2.0
    normalized_cutoff = cutoff / (input_rate * num_phases)

    # Centred at total_taps/2, so the group delay is exactly taps_per_phase/2
    # input samples (Kaiser's own window is centred at (N-1)/2 and would make
    # that a half sample).
    positions = np.arange(total_taps, dtype=np.float64) - total_taps / 2.0
    kernel = 2.0 * normalized_cutoff * np.sinc(2.0 * normalized_cutoff * positions)
    window_position = positions / (total_taps / 2.0)
    window = np.i0(_KAISER_BETA * np.sqrt(np.maximum(0.0, 1.0 - window_position**2)))
    kernel *= window / np.i0(_KAISER_BETA)

    # h[p + t*L] is branch p, tap t: reshape (taps, phases) and transpose.
    branches = kernel.reshape(taps_per_phase, num_phases).T.copy()
    # Each phase branch must have unit DC gain. This is also the xL
    # interpolation gain: without it every branch would sum to 1/L and the
    # output would be too quiet by L.
    branches /= branches.sum(axis=1, keepdims=True)
    coefficients = branches[:, ::-1].astype(np.float32)

    delay_output_samples = ceil(taps_per_phase * num_phases / (2.0 * step))
    return _Kernel(
        num_phases=num_phases,
        step=step,
        taps_per_phase=taps_per_phase,
        coefficients=coefficients,
        passband_hz=passband_hz,
        stopband_hz=stopband_hz,
        delay_output_samples=delay_output_samples,
    )


@lru_cache(maxsize=32)
def _cached_kernel(input_rate: int, output_rate: int) -> _Kernel:
    """Design once per rate pair; streams share the (read-only) coefficients."""
    return _design(input_rate, output_rate)


class PolyphaseResampler:
    """Streaming rational resampler with a declared ``kernel/2`` delay.

    ``push`` accepts any number of input samples and returns every output
    sample that can be computed now. The converter withholds its declared
    output delay at the head of the stream; the caller is expected to pay that
    with silence so the delivered stream keeps the input's length.
    """

    def __init__(self, input_rate: int, output_rate: int) -> None:
        self._kernel = _cached_kernel(input_rate, output_rate)
        self._tail = np.zeros(self._kernel.taps_per_phase - 1, dtype=np.float32)
        self._input_count = 0
        self._next_output = self._kernel.delay_output_samples

    @property
    def num_phases(self) -> int:
        """L in the reduced rate ratio."""
        return self._kernel.num_phases

    @property
    def step(self) -> int:
        """M in the reduced rate ratio."""
        return self._kernel.step

    @property
    def taps_per_phase(self) -> int:
        return self._kernel.taps_per_phase

    @property
    def delay_input_samples(self) -> int:
        """The converter's declared delay, in input samples (``kernel/2``)."""
        return self._kernel.delay_input_samples

    @property
    def delay_output_samples(self) -> int:
        """The declared delay expressed as the output samples withheld."""
        return self._kernel.delay_output_samples

    @property
    def passband_hz(self) -> float:
        return self._kernel.passband_hz

    @property
    def stopband_hz(self) -> float:
        return self._kernel.stopband_hz

    def reset(self) -> None:
        """Forget the stream. The next push starts a fresh, silent history."""
        self._tail = np.zeros(self._kernel.taps_per_phase - 1, dtype=np.float32)
        self._input_count = 0
        self._next_output = self._kernel.delay_output_samples

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Resample one block; return the output samples that are ready."""
        block = np.asarray(samples, dtype=np.float32)
        if block.ndim != 1:
            raise ValueError("PolyphaseResampler expects a 1-D mono block")
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)

        kernel = self._kernel
        taps = kernel.taps_per_phase
        buffer = np.concatenate((self._tail, block))
        total_input = self._input_count + block.size
        last_output = (total_input * kernel.num_phases - 1) // kernel.step

        if last_output < self._next_output:
            output = np.zeros(0, dtype=np.float32)
        else:
            indices = np.arange(self._next_output, last_output + 1, dtype=np.int64)
            phases = (indices * kernel.step) % kernel.num_phases
            input_positions = (indices * kernel.step - phases) // kernel.num_phases
            starts = input_positions - self._input_count
            if starts.size and starts.min() < 0:  # pragma: no cover - invariant
                raise RuntimeError("polyphase history underrun")
            windows = np.lib.stride_tricks.sliding_window_view(buffer, taps)[starts]
            output = np.einsum("ij,ij->i", windows, kernel.coefficients[phases])

        self._tail = buffer[-(taps - 1) :].copy()
        self._input_count = total_input
        # A push can be shorter than the converter's remaining warm-up; never
        # walk the next output index backwards in that case.
        self._next_output = max(self._next_output, last_output + 1)
        return output

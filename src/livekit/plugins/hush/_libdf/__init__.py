"""Pure-numpy replacement for the ``libdf`` C extension used by the Hush
plugin.

DeepFilterLib (the PyPI package the plugin used to depend on) is a
maturin-built Rust binding to the DeepFilterNet ``libdf`` library,
which provides a small set of DSP primitives:

  * ``DF.analysis`` / ``DF.synthesis`` — 50% overlap STFT/ISTFT with a
    Vorbis window and COLA scaling
  * ``erb`` — power-spectrogram projection onto an ERB-rate filterbank
  * ``erb_norm`` / ``unit_norm`` — exponential moving-average
    normalization (mean-subtract for ERB features, unit-norm for DF
    features)

The math is small (~200 lines of numpy) and not on any hot path that
would benefit from a C extension. Reimplementing it in numpy:

  * eliminates the Rust build dependency (pip can install without
    needing cargo on platforms without prebuilt wheels: Python 3.12+,
    musl libc, ARMv7, etc.)
  * matches the "no prebuilt mystery binaries" pitch in the plugin's
    README
  * makes every byte of the audio pipeline auditable Python in this
    repo

The implementation is verified to match ``libdf`` 0.5.6 bit-exactly
on round-trip analysis/synthesis and ERB/DF feature extraction (max
abs diff ~1e-7, just float32 numerical noise). See
``scripts/verify_against_pytorch.py`` for the end-to-end PyTorch
parity check.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Vorbis window: sin(pi/2 * sin^2(pi*(n+0.5)/N)) — matches the libdf
# `vorbis_window` function. Symmetric (length N), used for the STFT
# with 50% overlap.
def _vorbis_window(N: int) -> np.ndarray:
    n = np.arange(N, dtype=np.float64)
    sin_sq = np.sin(0.5 * np.pi * (n + 0.5) / (N // 2)) ** 2
    return np.sin(0.5 * np.pi * sin_sq).astype(np.float32)


# Window normalization: 1 / (window_size^2 / (2 * frame_size))
# = 2 * frame_size / window_size^2. Empirically verified against libdf.
def _compute_wnorm(fft_size: int, hop_size: int) -> float:
    return 1.0 / (fft_size ** 2 / (2.0 * hop_size))


# ERB feature normalization constants (from libdf `MEAN_NORM_INIT`).
MEAN_NORM_MIN = -60.0
MEAN_NORM_MAX = -90.0

# DF feature normalization constants (from libdf `UNIT_NORM_INIT`).
UNIT_NORM_MIN = 0.001
UNIT_NORM_MAX = 0.0001

# ERB projection floor (matches libdf's 1e-10 dB clamp).
ERB_FLOOR = 1e-10


# ---------------------------------------------------------------------------
# STFT/ISTFT
# ---------------------------------------------------------------------------


class DF:
    """Pure-numpy replacement for ``libdf.DF``.

    Same public API: ``analysis(audio, reset=False)`` returns the
    complex spectrum of shape ``(channels, n_frames, freq_bins)``;
    ``synthesis(spec, reset=False)`` returns the time-domain audio
    of shape ``(channels, n_frames * hop_size)``.
    """

    def __init__(
        self,
        sr: int,
        fft_size: int,
        hop_size: int,
        nb_bands: int,
        min_nb_erb_freqs: int = 1,
    ) -> None:
        assert hop_size * 2 <= fft_size, "hop_size must be <= fft_size / 2"
        self.sr = sr
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.freq_bins = fft_size // 2 + 1
        self.window = _vorbis_window(fft_size)
        self.wnorm = _compute_wnorm(fft_size, hop_size)
        # Ring buffer for the "context" between calls.
        self._analysis_mem = np.zeros(fft_size - hop_size, dtype=np.float32)
        self._synthesis_mem = np.zeros(fft_size - hop_size, dtype=np.float32)
        # ERB filterbank widths, computed via the libdf algorithm.
        self._erb_widths = _compute_erb_widths(
            sr, fft_size, nb_bands, min_nb_erb_freqs
        )
        # EMA state for the per-band mean normalization (ERB features).
        # Per-channel, shape (C, nb_bands). Initialized lazily by
        # ``erb_norm`` (or by the first call that uses it).
        self._erb_norm_state: np.ndarray | None = None
        self._unit_norm_state: np.ndarray | None = None

    def erb_widths(self) -> np.ndarray:
        return self._erb_widths

    def reset(self) -> None:
        self._analysis_mem[:] = 0
        self._synthesis_mem[:] = 0
        self._erb_norm_state = None
        self._unit_norm_state = None

    def analysis(self, audio: np.ndarray, reset: bool = False) -> np.ndarray:
        """STFT analysis. ``audio`` shape ``(channels, samples)``.

        Returns complex spectrogram of shape ``(channels, n_frames,
        freq_bins)`` where ``n_frames = samples // hop_size``.
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        n_channels, n_samples = audio.shape
        n_frames = n_samples // self.hop_size
        if reset:
            self._analysis_mem[:] = 0

        if n_frames == 0:
            return np.zeros(
                (n_channels, 0, self.freq_bins), dtype=np.complex64
            )

        output = np.empty(
            (n_channels, n_frames, self.freq_bins), dtype=np.complex64
        )
        for ch in range(n_channels):
            mem = self._analysis_mem.copy()
            for i in range(n_frames):
                chunk = audio[ch, i * self.hop_size : (i + 1) * self.hop_size]
                # Concatenate the previous context with the new input.
                buf = np.concatenate([mem, chunk])
                # Apply the Vorbis window and FFT, scaled by wnorm.
                spec = np.fft.rfft(buf * self.window) * self.wnorm
                output[ch, i] = spec.astype(np.complex64)
                # Update the context: for fft_size = 2 * hop_size, the
                # new context is just the new input chunk.
                if self.fft_size == 2 * self.hop_size:
                    mem = chunk.copy()
                else:
                    # General case: shift left by hop_size, append new.
                    mem = np.concatenate([mem[self.hop_size:], chunk])
            # Persist the per-channel context back to the shared buffer.
            self._analysis_mem = mem
        return output

    def synthesis(self, spec: np.ndarray, reset: bool = False) -> np.ndarray:
        """STFT synthesis. ``spec`` shape ``(channels, n_frames, freq_bins)``.

        Returns real audio of shape ``(channels, n_frames * hop_size)``.
        """
        if spec.ndim == 2:
            spec = spec[np.newaxis, :]
        spec = np.ascontiguousarray(spec, dtype=np.complex64)
        n_channels, n_frames, _ = spec.shape
        out_samples = n_frames * self.hop_size
        if reset:
            self._synthesis_mem[:] = 0

        if n_frames == 0:
            return np.zeros((n_channels, 0), dtype=np.float32)

        output = np.empty((n_channels, out_samples), dtype=np.float32)
        for ch in range(n_channels):
            mem = self._synthesis_mem.copy()
            for i in range(n_frames):
                # IFFT to time domain. N is built into the spec.
                x = np.fft.irfft(spec[ch, i], n=self.fft_size).astype(np.float32)
                # The irfft in numpy has a 1/N normalization, while
                # libdf's inverse FFT has none. Multiply by N to match.
                x *= self.fft_size
                # Apply the window (matches libdf's `apply_window_in_place`).
                x *= self.window
                # First hop_size samples: add to context, write to output.
                output[ch, i * self.hop_size : (i + 1) * self.hop_size] = (
                    x[: self.hop_size] + mem
                )
                # Update the context: for fft_size = 2 * hop_size, it's
                # just the second half of the windowed IFFT.
                if self.fft_size == 2 * self.hop_size:
                    mem = x[self.hop_size :].copy()
                else:
                    mem = x[self.hop_size :]
            self._synthesis_mem = mem
        return output


# ---------------------------------------------------------------------------
# ERB projection and exponential moving-average normalizations
# ---------------------------------------------------------------------------


def _compute_erb_widths(
    sr: int, fft_size: int, nb_bands: int, min_nb_freqs: int
) -> np.ndarray:
    """Reimplementation of libdf's ERB-width allocation.

    Matches the output of libdf's ``erb_fb`` (in Rust) for the default
    DeepFilterNet config (sr=16000, fft_size=320, nb_bands=32,
    min_nb_erb_freqs=2):

        array([ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                3, 6, 7, 7, 8, 8, 10, 12, 12, 14, 16, 18])

    Uses the Glasberg & Moore ERB-rate formula
    (``9.265 * ln(1 + f / 228.865)``) — the same one libdf uses, not
    the "Slaney" formula ``21.4 * log10(1 + f/229)`` that's in the
    PyTorch reference.
    """
    nyq_freq = sr / 2
    freq_width = float(sr) / fft_size

    def freq2erb(f: float) -> float:
        return 9.265 * np.log(1.0 + f / (24.7 * 9.265))

    def erb2freq(n: float) -> float:
        return 24.7 * 9.265 * (np.exp(n / 9.265) - 1.0)

    erb_low = freq2erb(0.0)
    erb_high = freq2erb(float(nyq_freq))
    step = (erb_high - erb_low) / nb_bands

    widths = [0] * nb_bands
    prev_freq = 0
    freq_over = 0
    for i in range(1, nb_bands + 1):
        f = erb2freq(erb_low + i * step)
        fb = int(f / freq_width + 0.5)
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        widths[i - 1] = nb_freqs
        prev_freq = fb
    # Account for the extra frequency bin (freq_size = fft_size/2 + 1
    # includes the DC and Nyquist bins).
    widths[nb_bands - 1] += 1
    too_large = sum(widths) - (fft_size // 2 + 1)
    if too_large > 0:
        widths[nb_bands - 1] -= too_large
    return np.array(widths, dtype=np.int64)


def erb(spec: np.ndarray, widths: np.ndarray, db: bool = True) -> np.ndarray:
    """ERB-rate projection of a complex spectrogram.

    ``spec`` shape: ``(..., freq_bins)`` (complex). ``widths`` is the
    ERB filterbank widths (sums to ``freq_bins``). Returns a real
    array of shape ``(..., nb_bands)``. If ``db`` is True (default),
    output is in decibels (``10 * log10(mean_power + 1e-10)``).

    Matches libdf's ``compute_band_corr``: each ERB band is the
    *mean* of the squared magnitudes of its bins, not the sum.
    """
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
    flat = power.reshape(-1, power.shape[-1])
    out = np.empty((flat.shape[0], len(widths)), dtype=np.float32)
    start = 0
    for b, w in enumerate(widths):
        out[:, b] = flat[:, start : start + w].mean(axis=1)
        start += int(w)
    if db:
        # Libdf: out = 10 * log10(out + 1e-10) — adds the floor to the
        # value, not clip-then-log.
        out += ERB_FLOOR
        np.log10(out, out=out)
        out *= 10.0
    return out.reshape(*power.shape[:-1], len(widths))


def erb_norm(
    x: np.ndarray, alpha: float, state: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exponential moving-average mean-subtraction for ERB features.

    Matches libdf's ``band_mean_norm_erb``: per-bin, the state is
    updated as ``s = x*(1-alpha) + s*alpha`` and the output is
    ``(x - s) / 40``.

    The state is per-channel, shape ``(C, E)``, and persists across
    time steps within each channel. If ``state`` is ``None``, it is
    initialized to ``linspace(MEAN_NORM_MIN, MEAN_NORM_MAX, E)``.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    E = x.shape[-1]
    C = x.shape[0] if x.ndim >= 2 else 1
    T = x.shape[1] if x.ndim >= 3 else 1
    if state is None:
        state = np.broadcast_to(
            np.linspace(MEAN_NORM_MIN, MEAN_NORM_MAX, E, dtype=np.float32),
            (C, E),
        ).copy()
    else:
        state = np.ascontiguousarray(state, dtype=np.float32)
    # Iterate: for c in C, for t in T, update s[c] in place using x[c, t]
    # Match libdf's exact float32 arithmetic: cast (1-a) to f32 so the
    # subtraction happens in f32 (not f64) and matches libdf's Rust code.
    one_minus_alpha = np.float32(1.0) - np.float32(alpha)
    for c in range(C):
        s = state[c]
        for t in range(T):
            row = x[c, t]
            s *= alpha
            s += row * one_minus_alpha
            x[c, t] = (row - s) / 40.0
    return x, state


def unit_norm(
    spec: np.ndarray, alpha: float, state: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exponential moving-average unit-norm for DF features.

    Matches libdf's ``band_unit_norm``: per-bin, the state is
    updated as ``s = |x|*(1-alpha) + s*alpha`` and the output is
    ``x / sqrt(s)``.

    The state is per-channel, shape ``(C, F)``, and persists across
    time steps within each channel. If ``state`` is ``None``, it is
    initialized to ``linspace(UNIT_NORM_MIN, UNIT_NORM_MAX, F)``.
    """
    spec = np.ascontiguousarray(spec, dtype=np.complex64)
    F = spec.shape[-1]
    C = spec.shape[0] if spec.ndim >= 2 else 1
    T = spec.shape[1] if spec.ndim >= 3 else 1
    if state is None:
        state = np.broadcast_to(
            np.linspace(UNIT_NORM_MIN, UNIT_NORM_MAX, F, dtype=np.float32),
            (C, F),
        ).copy()
    else:
        state = np.ascontiguousarray(state, dtype=np.float32)
    # Match libdf's exact float32 arithmetic: cast (1-a) to f32 so the
    # subtraction happens in f32 (not f64) and matches libdf's Rust code.
    one_minus_alpha = np.float32(1.0) - np.float32(alpha)
    for c in range(C):
        s = state[c]
        for t in range(T):
            row = spec[c, t]
            s *= alpha
            s += np.abs(row) * one_minus_alpha
            spec[c, t] = row / np.sqrt(s)
    return spec, state

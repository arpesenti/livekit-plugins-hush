"""Compare the in-repo polyphase resampler with the converter it replaced.

The production path no longer uses ``rtc.AudioResampler``. This script keeps a
copy of the old behaviour only as a reference: it runs ``docs/audio/originals/
speech.wav`` through both converters and reports how far their spectra differ
in the speech band, plus the stopband/image rejection that the old black box
did not document.

Stated tolerance (the script exits non-zero outside it):

* passband spectral difference, 100 Hz to the lower of 7 kHz / the converter's
  passband edge: mean <= 0.10 dB, worst bin <= 0.50 dB;
* anti-alias rejection of a tone 300 Hz above the stopband edge: >= 60 dB;
* interpolation image rejection for a 1 kHz tone: >= 60 dB.

Measured on this checkout (2026-11): mean 0.00-0.02 dB, worst 0.15 dB, and
77-95 dB of stopband/image rejection. The 7.2-8.0 kHz roll-off is deliberate
and is where the current converter and the new one differ most; it is the
anti-alias edge, not a broadband artefact.

    python scripts/compare_resampler_quality.py

Dependencies: numpy, livekit-rtc (already required by the plugin).
"""

import os
import sys
import wave

import numpy as np

from livekit import rtc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from livekit.plugins.hush._resampler import PolyphaseResampler

PASSBAND_MEAN_DB = 0.10
PASSBAND_MAX_DB = 0.50
STOPBAND_MIN_DB = 60.0
RATES = (8000, 16000, 24000, 32000, 44100, 48000)


def _load_speech() -> np.ndarray:
    path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "audio", "originals", "speech.wav"
    )
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width != 2 or rate != 16000:
        raise SystemExit(
            f"expected 16-bit 16 kHz speech.wav, got {width * 8}-bit {rate} Hz"
        )
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def _old_convert(
    input_rate: int, output_rate: int, audio: np.ndarray, frame_ms: int = 10
) -> np.ndarray:
    resampler = rtc.AudioResampler(
        input_rate=input_rate,
        output_rate=output_rate,
        num_channels=1,
        quality=rtc.AudioResamplerQuality.MEDIUM,
    )
    n = int(input_rate * frame_ms / 1000)
    blocks = []
    for i in range(0, len(audio) - n + 1, n):
        pcm = (np.clip(audio[i : i + n], -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = rtc.AudioFrame(
            data=pcm.tobytes(),
            sample_rate=input_rate,
            num_channels=1,
            samples_per_channel=n,
        )
        for out in resampler.push(frame):
            blocks.append(np.frombuffer(out.data, dtype=np.int16) / 32768.0)
    return np.concatenate(blocks) if blocks else np.zeros(0)


def _new_convert(
    input_rate: int, output_rate: int, audio: np.ndarray, frame_ms: int = 10
) -> np.ndarray:
    resampler = PolyphaseResampler(input_rate, output_rate)
    n = int(input_rate * frame_ms / 1000)
    blocks = [resampler.push(audio[i : i + n]) for i in range(0, len(audio) - n + 1, n)]
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


def _psd(signal: np.ndarray, rate: int, nfft: int = 2048):
    window = np.hanning(nfft)
    step = nfft // 2
    acc = np.zeros(nfft // 2 + 1)
    count = 0
    for i in range(0, len(signal) - nfft, step):
        acc += np.abs(np.fft.rfft(signal[i : i + nfft] * window)) ** 2
        count += 1
    return np.fft.rfftfreq(nfft, 1 / rate), acc / max(count, 1)


def _band_difference(
    old: np.ndarray, new: np.ndarray, rate: int, low: float, high: float
) -> tuple[float, float]:
    n = min(len(old), len(new))
    freq, old_psd = _psd(old[:n], rate)
    _, new_psd = _psd(new[:n], rate)
    selected = (freq >= low) & (freq <= high)
    difference = 10 * np.log10(
        np.maximum(new_psd[selected], 1e-30) / np.maximum(old_psd[selected], 1e-30)
    )
    return float(np.mean(np.abs(difference))), float(np.max(np.abs(difference)))


def _tone_amplitude(signal: np.ndarray, rate: int, frequency: float) -> float:
    if len(signal) == 0:
        return 0.0
    t = np.arange(len(signal)) / rate
    return float(
        2.0 * abs(np.sum(signal * np.exp(-2j * np.pi * frequency * t))) / len(signal)
    )


def _fold(frequency: float, rate: int) -> float:
    return abs(frequency - rate * round(frequency / rate))


def _tone_test(
    input_rate: int,
    output_rate: int,
    frequency: float,
    measure: float | None = None,
) -> float:
    """Rejection of `frequency` in dB, measured where `measure` folds to."""
    resampler = PolyphaseResampler(input_rate, output_rate)
    t = np.arange(input_rate) / input_rate
    tone = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    n = input_rate // 100
    out = np.concatenate(
        [resampler.push(tone[i : i + n]) for i in range(0, len(tone), n)]
    )
    measured = _fold(frequency if measure is None else measure, output_rate)
    amplitude = _tone_amplitude(out[output_rate // 2 :], output_rate, measured)
    return 20 * np.log10(max(amplitude, 1e-12) / 0.5)


def main() -> None:
    speech = _load_speech()
    print(f"reference: speech.wav, {len(speech) / 16000:.2f} s")
    print(
        f"tolerance: passband mean <= {PASSBAND_MEAN_DB:.2f} dB, "
        f"worst bin <= {PASSBAND_MAX_DB:.2f} dB\n"
    )

    failures = []
    print("| Conversion | Passband | Mean | Worst |")
    print("|---|---|---|---|")
    for rate in RATES:
        if rate == 16000:
            continue
        passband_high = min(7000.0, PolyphaseResampler(16000, rate).passband_hz - 200.0)
        if rate < 16000:
            passband_high = min(
                passband_high, PolyphaseResampler(rate, 16000).passband_hz - 200.0
            )

        source_up = _old_convert(16000, rate, speech)
        new_up = _new_convert(16000, rate, speech)
        mean_up, max_up = _band_difference(
            source_up, new_up, rate, 100.0, passband_high
        )

        old_down = _old_convert(rate, 16000, source_up)
        new_down = _new_convert(rate, 16000, source_up)
        mean_down, max_down = _band_difference(
            old_down,
            new_down,
            16000,
            100.0,
            min(7000.0, 3000.0 if rate == 8000 else 7000.0),
        )

        for direction, mean, worst in (
            (f"{rate}->16000", mean_down, max_down),
            (f"16000->{rate}", mean_up, max_up),
        ):
            print(
                f"| {direction} | 100-{passband_high:.0f} Hz | {mean:.2f} dB | {worst:.2f} dB |"
            )
            if mean > PASSBAND_MEAN_DB or worst > PASSBAND_MAX_DB:
                failures.append(f"{direction}: {mean:.2f} / {worst:.2f} dB")

        if rate < 16000:
            # 8 kHz interpolates on the way in and decimates on the way out.
            # A stopband tone above the input Nyquist is not representable, so
            # the meaningful interpolation check is the 1 kHz image.
            image = _tone_test(8000, 16000, 1000.0, measure=8000.0 - 1000.0)
            label = "image 1 kHz (8000->16000)"
            print(f"| {label} | | rejection {image:.1f} dB | |")
            if image > -STOPBAND_MIN_DB:
                failures.append(f"{label}: {image:.1f} dB")

            stopband = PolyphaseResampler(16000, 8000).stopband_hz
            rejection = _tone_test(16000, 8000, stopband + 300.0)
            label = f"stopband {stopband + 300.0:.0f} Hz (16000->8000)"
            print(f"| {label} | | rejection {rejection:.1f} dB | |")
            if rejection > -STOPBAND_MIN_DB:
                failures.append(f"{label}: {rejection:.1f} dB")
        else:
            stopband = PolyphaseResampler(rate, 16000).stopband_hz
            rejection = _tone_test(rate, 16000, stopband + 300.0)
            label = f"stopband {stopband + 300.0:.0f} Hz ({rate}->16000)"
            print(f"| {label} | | rejection {rejection:.1f} dB | |")
            if rejection > -STOPBAND_MIN_DB:
                failures.append(f"{label}: {rejection:.1f} dB")

            # A 16 kHz input cannot carry a tone above 8 kHz; interpolation
            # aliasing is the 1 kHz image at 15 kHz, folded to the output rate.
            image = _tone_test(16000, rate, 1000.0, measure=16000.0 - 1000.0)
            label = f"image 1 kHz (16000->{rate})"
            print(f"| {label} | | rejection {image:.1f} dB | |")
            if image > -STOPBAND_MIN_DB:
                failures.append(f"{label}: {image:.1f} dB")

    print()
    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("PASS: passband parity and anti-alias rejection within the stated tolerance")


if __name__ == "__main__":
    main()

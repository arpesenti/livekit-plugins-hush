"""Measure the delay the enhanced path adds, end to end.

Cross-correlates the stage's input against its output for every rate and frame
size in the acceptance matrix. The default hop processor is a plain gain, so
the number is the rate conversion + hop/frame quantisation, not the model's own
10 ms STFT lookahead. Pass ``--model`` to drive the real ONNX model as well
(slower; its algorithmic latency is included in that run).

This is the script behind the added-delay table in
``docs/adr/0001-no-leak-enhanced-path.md`` and the "Added delay" section of the
README. Re-run it after touching the converters or the prime.

    python scripts/measure_delay.py
    python scripts/measure_delay.py --seconds 30
    python scripts/measure_delay.py --model

Dependencies: numpy, and the plugin's own dependencies for ``--model``.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RATES = (8000, 16000, 24000, 32000, 44100, 48000)
FRAME_MS = (10, 20, 50, 100)


def _lag(output: np.ndarray, source: np.ndarray, max_lag: int) -> int:
    """Delay of `output` behind `source`, in samples (both at the same rate)."""
    n = min(len(output), len(source))
    output = output[:n].astype(np.float64)
    source = source[:n].astype(np.float64)
    output -= output.mean()
    source -= source.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    corr = np.fft.irfft(
        np.fft.rfft(output, nfft) * np.conj(np.fft.rfft(source, nfft)), nfft
    )
    return int(np.argmax(corr[: max_lag + 1]))


def _model_processor():
    from livekit.plugins.hush import HushNoiseSuppressor

    suppressor = HushNoiseSuppressor(strength=1.0)

    def process(hop: np.ndarray) -> np.ndarray:
        return suppressor._process_hop(hop)

    return process, suppressor


def _gain_processor():
    def process(hop: np.ndarray) -> np.ndarray:
        return hop * 0.5

    return process, None


def _run(rate: int, frame_ms: int, seconds: float, use_model: bool):
    from livekit.plugins.hush.noise_suppressor import _StreamStage

    process_hop, suppressor = _model_processor() if use_model else _gain_processor()
    stage = _StreamStage(strength=1.0, process_hop=process_hop)
    stage.rebuild(rate)

    n = int(rate * frame_ms / 1000)
    blocks = int(seconds * 1000 / frame_ms)
    if use_model:
        # A denoiser removes most of a white-noise probe; a chirp stays
        # broadband and audible to it, so the correlation has a real peak.
        t = np.arange(n * blocks) / rate
        top = min(3000.0, rate * 0.4)
        sweep = 2 * np.pi * (100.0 * t + (top - 100.0) / (2 * seconds) * t**2)
        signal = (0.2 * np.sin(sweep)).astype(np.float32)
    else:
        rng = np.random.default_rng(0)
        signal = (rng.standard_normal(n * blocks) * 0.1).astype(np.float32)

    source, output = [], []
    for i in range(blocks):
        block = signal[i * n : (i + 1) * n]
        source.append(block)
        output.append(stage.push(block))

    src = np.concatenate(source)
    out = np.concatenate(output)
    lag = _lag(out, src, max_lag=rate // 2)
    result = {
        "lag_ms": lag / rate * 1000.0,
        "prime_ms": stage.prime_samples / rate * 1000.0,
        "declared_ms": stage.declared_delay_samples / rate * 1000.0,
        "hops": stage.hops_processed,
        "underruns": stage.underruns,
        "failures": stage.failures,
    }
    if suppressor is not None:
        suppressor._close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds", type=float, default=20.0, help="audio per combination"
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="drive the real ONNX model instead of a gain (includes its 10 ms)",
    )
    args = parser.parse_args()

    print(
        f"Added delay {'with the real model (includes ~10 ms algorithm)' if args.model else 'with a stand-in hop processor (algorithm excluded)'}"
    )
    print(f"{args.seconds:g} s per combination, cross-correlation lag\n")
    header = "| In rate | " + " | ".join(f"{ms} ms" for ms in FRAME_MS) + " |"
    print(header)
    print("|---|" + "---|" * len(FRAME_MS))

    failures = 0
    for rate in RATES:
        cells = []
        for frame_ms in FRAME_MS:
            r = _run(rate, frame_ms, args.seconds, args.model)
            cells.append(f"{r['lag_ms']:.1f} ms")
            if r["underruns"] or r["failures"]:
                failures += 1
                cells[-1] += f" (underruns={r['underruns']}, failures={r['failures']})"
        print(f"| {rate} | " + " | ".join(cells) + " |")

    print("\nPer-combination detail (prime and declared converter delay):\n")
    print("| In rate | Frame | Lag | Prime | Declared | Hops | Underruns |")
    print("|---|---|---|---|---|---|---|")
    for rate in RATES:
        for frame_ms in FRAME_MS:
            r = _run(rate, frame_ms, args.seconds, args.model)
            print(
                f"| {rate} | {frame_ms} ms | {r['lag_ms']:.1f} ms | "
                f"{r['prime_ms']:.1f} ms | {r['declared_ms']:.1f} ms | "
                f"{r['hops']} | {r['underruns']} |"
            )

    if failures:
        print(f"\nFAIL: {failures} combinations reported underruns or failures")
        sys.exit(1)


if __name__ == "__main__":
    main()

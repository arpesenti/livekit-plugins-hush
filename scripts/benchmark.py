"""Benchmark the Hush plugin's per-frame latency and multi-stream throughput.

Measures:
  1. Single-stream latency — how long ``HushSession.process_frame`` takes
     per 10 ms frame (the per-call CPU cost in the LiveKit FrameProcessor).
  2. Multi-stream throughput — how the plugin scales when many sessions
     run concurrently on the same machine (the typical agent workload:
     N concurrent calls in one worker process, all competing for cores).
  3. DSP-only cost — the pure-numpy frontend alone (STFT/ISTFT + ERB +
     per-band EMA), useful for comparing to the ORT-inference cost.

All benchmarks use ``time.perf_counter_ns`` and a per-frame ``gc.disable``
window to suppress GC pauses in the measurement.

Run:

    python scripts/benchmark.py
    python scripts/benchmark.py --help

Output is printed as a small table; pass ``--json out.json`` to also
write machine-readable results.

Dependencies: numpy, onnxruntime (both already required by the plugin).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def _system_info() -> dict:
    """Collect machine/CPython/NumPy info for the benchmark header."""
    import onnxruntime as ort
    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "cpu_count": os.cpu_count(),
    }
    # Best-effort: report the ORT build's CPU features so results from
    # different machines can be compared apples-to-apples.
    try:
        info["ort_cpu_features"] = ort.get_available_providers()
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------


def _load_speech(seconds: float = 10.0) -> np.ndarray:
    """Load ``docs/audio/originals/speech.wav`` and return float32 mono.

    Falls back to a synthesized sine sweep if the file isn't present
    (so the benchmark still runs in a fresh checkout).
    """
    candidates = [
        os.path.join(
            os.path.dirname(__file__), "..", "docs", "audio", "originals", "speech.wav"
        ),
        "/home/brains99/livekit-plugins-hush/docs/audio/originals/speech.wav",
    ]
    for path in candidates:
        if os.path.exists(path):
            import wave
            with wave.open(path, "rb") as wf:
                n = min(int(wf.getframerate() * seconds), wf.getnframes())
                audio = (
                    np.frombuffer(wf.readframes(n), dtype=np.int16)
                    .astype(np.float32)
                    / 32768.0
                )
            return audio
    # Fallback: synthetic 440 Hz + 1 kHz mix.
    sr = 16000
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1000 * t)).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Single-stream timing
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    """Per-component wall-clock timings for a chunk of frames.

    All times are in milliseconds. ``frames`` is the number of 10 ms
    frames that were processed; ``elapsed_ms`` is the wall-clock time
    for the chunk. ``per_frame_ms`` is ``elapsed_ms / frames``.
    """

    name: str
    frames: int
    elapsed_ms: float
    per_frame_us: float  # microseconds per frame
    std_us: float  # standard deviation of per-frame time, in microseconds
    realtime_factor: float  # 10.0 / per_frame_ms; >1 means real-time

    def as_row(self) -> str:
        return (
            f"  {self.name:<32s} "
            f"{self.per_frame_us:7.1f} µs/frame  "
            f"±{self.std_us:5.1f} µs  "
            f"({self.realtime_factor:5.1f}x real-time)"
        )

def _bench_single_stream(
    audio: np.ndarray, n_warmup: int = 50, n_frames: int = 1000, n_trials: int = 5
) -> list[TimingResult]:
    """Run the full per-frame pipeline and time it.

    Returns one TimingResult per trial; the caller can take the median.
    """
    from livekit.plugins.hush._hush_model import HushModel, HushSession

    model = HushModel()
    session = HushSession(model)

    # Slice the audio into 160-sample chunks. If the input is too short,
    # truncate the run rather than failing partway through.
    available_frames = len(audio) // 160
    total_needed = n_warmup + n_frames
    if available_frames < total_needed:
        shortfall = total_needed - available_frames
        print(
            f"  warning: audio has {available_frames} frames but "
            f"{total_needed} needed; trimming measured run to "
            f"{max(0, available_frames - n_warmup)} frames"
        )
        n_frames = max(0, available_frames - n_warmup)
        if n_frames <= 0:
            session.close()
            return []

    chunks = [
        audio[i * 160 : (i + 1) * 160].copy()
        for i in range(n_warmup + n_frames)
    ]

    # Warmup.
    for i in range(n_warmup):
        session.process_frame(chunks[i])

    results: list[TimingResult] = []
    for trial in range(n_trials):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter_ns()
            for i in range(n_warmup, n_warmup + n_frames):
                session.process_frame(chunks[i])
            t1 = time.perf_counter_ns()
        finally:
            gc.enable()

        elapsed_ms = (t1 - t0) / 1e6
        per_frame_us = elapsed_ms * 1000.0 / n_frames
        realtime_factor = 10.0 / per_frame_us * 1000.0  # 10ms budget per frame
        results.append(
            TimingResult(
                name=f"single-stream trial {trial + 1}",
                frames=n_frames,
                elapsed_ms=elapsed_ms,
                per_frame_us=per_frame_us,
                std_us=0.0,  # We don't have per-call samples; std is 0.
                realtime_factor=realtime_factor,
            )
        )
    session.close()
    return results


# ---------------------------------------------------------------------------
# Multi-stream throughput
# ---------------------------------------------------------------------------


@dataclass
class MultiStreamResult:
    """Throughput at a given concurrency level.

    For N concurrent sessions, this is the wall-clock time to process
    one full chunk of frames (N_TOTAL_FRAMES / N frames per session),
    with all N sessions advancing in lockstep via a thread pool.
    """

    concurrency: int
    frames_per_session: int
    total_frames: int
    elapsed_ms: float
    per_frame_us_avg: float  # avg µs to process one frame across all streams
    cpu_pct_of_one_core: float  # (elapsed_ms / (frames * 10ms)) * 100

    def as_row(self) -> str:
        return (
            f"  N={self.concurrency:<3d} streams  "
            f"{self.per_frame_us_avg:7.1f} µs/frame (avg)  "
            f"elapsed={self.elapsed_ms:7.1f} ms  "
            f"≈{self.cpu_pct_of_one_core:5.1f}% of one core"
        )


def _bench_multi_stream(
    audio: np.ndarray,
    n_streams: list[int],
    n_warmup: int = 30,
    n_frames_per_session: int = 500,
) -> list[MultiStreamResult]:
    """Time N concurrent sessions, each processing a 10 ms frame at a time.

    Uses a thread pool to advance the streams in lockstep. This is the
    "many concurrent calls" agent workload: N parallel HushSessions
    competing for the same CPU.
    """
    from concurrent.futures import ThreadPoolExecutor
    from livekit.plugins.hush._hush_model import HushModel, HushSession

    model = HushModel()

    # If the audio is too short, trim the run.
    available_frames = len(audio) // 160
    total_needed = n_warmup + n_frames_per_session
    if available_frames < total_needed:
        n_frames_per_session = max(0, available_frames - n_warmup)
        if n_frames_per_session <= 0:
            return []

    chunks = [
        audio[i * 160 : (i + 1) * 160].copy()
        for i in range(n_warmup + n_frames_per_session)
    ]

    def make_session() -> HushSession:
        return HushSession(model)

    results: list[MultiStreamResult] = []
    for n in n_streams:
        sessions = [make_session() for _ in range(n)]

        def worker(sess: HushSession) -> None:
            for i in range(n_warmup + n_frames_per_session):
                sess.process_frame(chunks[i])

        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter_ns()
            with ThreadPoolExecutor(max_workers=n) as ex:
                list(ex.map(worker, sessions))
            t1 = time.perf_counter_ns()
        finally:
            gc.enable()

        for s in sessions:
            s.close()

        elapsed_ms = (t1 - t0) / 1e6
        total_frames = n * n_frames_per_session
        per_frame_us = elapsed_ms * 1000.0 / total_frames
        # CPU% of one core = (time spent) / (time budget) for the work
        # (n * n_frames * 10ms of audio needed in real-time).
        cpu_pct = elapsed_ms * 100.0 / (n_frames_per_session * 10.0)
        results.append(
            MultiStreamResult(
                concurrency=n,
                frames_per_session=n_frames_per_session,
                total_frames=total_frames,
                elapsed_ms=elapsed_ms,
                per_frame_us_avg=per_frame_us,
                cpu_pct_of_one_core=cpu_pct,
            )
        )
    return results


# ---------------------------------------------------------------------------
# DSP-only cost
# ---------------------------------------------------------------------------


def _bench_dsp_components(
    audio: np.ndarray, n_warmup: int = 50, n_frames: int = 1000
) -> list[TimingResult]:
    """Time the pure-numpy DSP primitives in isolation, separate from ORT.

    Useful to see how much of the per-frame cost is the STFT/ISTFT/ERB
    vs the three ONNX sub-model inferences.
    """
    from livekit.plugins.hush._libdf import DF, erb, erb_norm, unit_norm

    df = DF(sr=16000, fft_size=320, hop_size=160, nb_bands=32, min_nb_erb_freqs=2)
    alpha = 0.99

    # Trim the run if the audio is too short.
    available_frames = len(audio) // 160
    total_needed = n_warmup + n_frames
    if available_frames < total_needed:
        n_frames = max(0, available_frames - n_warmup)
        if n_frames <= 0:
            return []

    chunks = [audio[i * 160 : (i + 1) * 160].copy() for i in range(n_warmup + n_frames)]
    widths = df.erb_widths()

    def full_dsp():
        """One frame: analysis + erb + erb_norm + unit_norm + synthesis."""
        spec = df.analysis(chunks[0], reset=False)
        e, df._erb_norm_state = erb_norm(erb(spec, widths), alpha, df._erb_norm_state)
        f, df._unit_norm_state = unit_norm(spec[..., :64].copy(), alpha, df._unit_norm_state)
        return df.synthesis(spec, reset=False)

    # Warmup (also primes the state)
    for i in range(n_warmup):
        full_dsp()
        chunks_used = chunks[i]

    def time_loop(label: str, body):
        # body: callable that processes ONE frame using outer-scope chunks[i]
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter_ns()
            for i in range(n_warmup, n_warmup + n_frames):
                body(i)
            t1 = time.perf_counter_ns()
        finally:
            gc.enable()
        elapsed_ms = (t1 - t0) / 1e6
        per_frame_us = elapsed_ms * 1000.0 / n_frames
        realtime_factor = 10.0 / per_frame_us * 1000.0
        return TimingResult(
            name=label,
            frames=n_frames,
            elapsed_ms=elapsed_ms,
            per_frame_us=per_frame_us,
            std_us=0.0,
            realtime_factor=realtime_factor,
        )

    results: list[TimingResult] = []

    def analysis_only(i):
        df.analysis(chunks[i], reset=False)

    def analysis_plus_erb(i):
        spec = df.analysis(chunks[i], reset=False)
        erb(spec, widths)

    def analysis_plus_erb_plus_erbnorm(i):
        spec = df.analysis(chunks[i], reset=False)
        e, df._erb_norm_state = erb_norm(erb(spec, widths), alpha, df._erb_norm_state)
    # Alias for backward-readable label.
    analysis_plus_erbnorm = analysis_plus_erb_plus_erbnorm

    def analysis_plus_erbnorm_plus_unitnorm(i):
        spec = df.analysis(chunks[i], reset=False)
        e, df._erb_norm_state = erb_norm(erb(spec, widths), alpha, df._erb_norm_state)
        f, df._unit_norm_state = unit_norm(spec[..., :64].copy(), alpha, df._unit_norm_state)
    analysis_plus_unitnorm = analysis_plus_erbnorm_plus_unitnorm

    def full_dsp_loop(i):
        spec = df.analysis(chunks[i], reset=False)
        e, df._erb_norm_state = erb_norm(erb(spec, widths), alpha, df._erb_norm_state)
        f, df._unit_norm_state = unit_norm(spec[..., :64].copy(), alpha, df._unit_norm_state)
        df.synthesis(spec, reset=False)

    def synthesis_only(i):
        # Use a fresh spec to isolate the cost
        df.synthesis(
            np.zeros((1, 1, 161), dtype=np.complex64), reset=False
        )

    results.append(time_loop("analysis only (STFT)", analysis_only))
    results.append(
        time_loop("analysis + erb projection", analysis_plus_erb)
    )
    results.append(
        time_loop(
            "analysis + erb + erb_norm (per-band EMA)", analysis_plus_erb_plus_erbnorm
        )
    )
    results.append(
        time_loop(
            "analysis + erb_norm + unit_norm (full features)",
            analysis_plus_erbnorm_plus_unitnorm,
        )
    )
    results.append(time_loop("synthesis only (ISTFT)", synthesis_only))
    results.append(time_loop("full DSP pipeline", full_dsp_loop))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_system(info: dict) -> list[str]:
    lines = []
    lines.append("=" * 72)
    lines.append("Hush plugin benchmark")
    lines.append("=" * 72)
    lines.append(f"  Python:     {info['python']}")
    lines.append(f"  Platform:   {info['platform']}")
    lines.append(f"  Machine:    {info['machine']}")
    lines.append(f"  CPU:        {info['processor']}")
    lines.append(f"  CPU cores:  {info['cpu_count']}")
    lines.append(f"  NumPy:      {info['numpy']}")
    lines.append(f"  ORT:        {info['onnxruntime']}")
    if "ort_cpu_features" in info:
        lines.append(f"  ORT EPs:    {info['ort_cpu_features']}")
    lines.append("=" * 72)
    return lines


def _print_section(title: str, items: list[TimingResult]) -> None:
    print()
    print(title)
    print("-" * 72)
    for it in items:
        print(it.as_row())


def _print_multi(items: list[MultiStreamResult]) -> None:
    print()
    print("Multi-stream throughput (concurrent sessions on the same CPU)")
    print("-" * 72)
    for it in items:
        print(it.as_row())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=20.0,
        help=(
            "Length of synthesized/loaded audio in seconds. Must be at least "
            "(n_warmup + n_frames) * 0.01 + 1 (default: 20)"
        ),
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=1000,
        help="Frames to time per single-stream trial (default: 1000)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=50,
        help="Warmup frames before each measurement (default: 50)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Single-stream trials to run; the median is reported (default: 5)",
    )
    parser.add_argument(
        "--multi-streams",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated concurrency levels for multi-stream bench",
    )
    parser.add_argument(
        "--multi-frames",
        type=int,
        default=500,
        help="Frames per session in the multi-stream bench (default: 500)",
    )
    parser.add_argument(
        "--skip-multi",
        action="store_true",
        help="Skip the multi-stream benchmark (saves time on slow systems)",
    )
    parser.add_argument(
        "--skip-dsp",
        action="store_true",
        help="Skip the DSP-component breakdown",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Write machine-readable results to this JSON file",
    )
    args = parser.parse_args()

    info = _system_info()
    for line in _format_system(info):
        print(line)

    audio = _load_speech(args.audio_seconds)
    print(f"\nAudio fixture: {len(audio) / 16000:.1f}s at 16 kHz")

    json_out: dict = {"system": info, "single_stream": [], "multi_stream": [], "dsp_components": []}

    # 1. Single-stream latency
    single = _bench_single_stream(
        audio,
        n_warmup=args.n_warmup,
        n_frames=args.n_frames,
        n_trials=args.n_trials,
    )
    median_idx = sorted(range(len(single)), key=lambda i: single[i].per_frame_us)[
        len(single) // 2
    ]
    median = single[median_idx]
    # Replace the trial name with a cleaner label for the median line.
    median.name = "single-stream (median of trials)"
    _print_section(
        "Single-stream latency (per-frame, full pipeline incl. ORT)", single
    )
    print(
        f"\n  → Median: {median.per_frame_us:.1f} µs/frame "
        f"({median.realtime_factor:.1f}x real-time headroom on one core)"
    )
    for r in single:
        json_out["single_stream"].append(asdict(r))

    # 2. Multi-stream throughput
    if not args.skip_multi:
        try:
            levels = [int(x) for x in args.multi_streams.split(",") if x.strip()]
        except ValueError:
            print("Invalid --multi-streams; expected comma-separated ints")
            return 2
        multi = _bench_multi_stream(
            audio, levels, n_warmup=args.n_warmup, n_frames_per_session=args.multi_frames
        )
        _print_multi(multi)
        for r in multi:
            json_out["multi_stream"].append(asdict(r))

    # 3. DSP component breakdown
    if not args.skip_dsp:
        dsp = _bench_dsp_components(
            audio, n_warmup=args.n_warmup, n_frames=args.n_frames
        )
        _print_section("Pure-numpy DSP components (no ORT)", dsp)
        for r in dsp:
            json_out["dsp_components"].append(asdict(r))

    print()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(json_out, f, indent=2)
        print(f"Results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

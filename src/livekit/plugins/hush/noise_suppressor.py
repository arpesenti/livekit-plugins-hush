"""Hush noise suppression as a LiveKit FrameProcessor.

Implements the Hush speech enhancement model from pulp-vision/Hush:
  Built on DeepFilterNet3 with a novel auxiliary separation head that
  teaches the encoder to distinguish speakers — not just speech from noise.
  Trained on 10,000+ hours of mixed audio with 60% of samples including
  a competing human speaker.

Uses DeepFilterLib (open-source C library) for ERB feature extraction and
ONNX Runtime for inference — no PyTorch, no prebuilt mystery binaries.

Model sharing: the ONNX session is loaded once per worker process and
shared across all instances. Each ``HushNoiseSuppressor`` holds its own
per-stream DF state and audio buffers.

Memory per worker: ~40 MB (onnxruntime + model) + N × ~1 MB (per session).

Drop-in replacement for livekit-plugins-noise-cancellation / ai-coustics.

Streaming contract: a frame in always produces a frame of the same length
out, drawn from the enhanced path. The frame rate the host uses and the
model's 160-sample hop are decoupled by ``_StreamStage``, which buffers,
resamples and — at the head of a stream only — emits silence so the model
never has to be bypassed. See ``docs/adr/0001-no-leak-enhanced-path.md``.
"""

import logging
import os
from collections.abc import Callable

import numpy as np

from livekit import rtc

from ._hush_model import (
    _FRAME_SAMPLES,
    _SAMPLE_RATE,
    HushSession,
    _get_shared_model,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

#: Converts one 160-sample hop at 16 kHz into enhanced audio.
HopProcessor = Callable[[np.ndarray], np.ndarray]


def _frame_to_float(frame: rtc.AudioFrame) -> np.ndarray:
    return np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0


def _to_int16_frame(samples: np.ndarray, sample_rate: int) -> rtc.AudioFrame:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    return rtc.AudioFrame(
        data=pcm.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=len(pcm),
    )


def _measure_backlog(rate: int, seconds: float = 0.5) -> int:
    """Native-rate samples the converter pair can hold back at its worst.

    A streaming resampler buffers before it returns anything, and the amount
    it holds oscillates over its burst cycle. The peak is the dead time the
    stage cannot see through its own FIFO: sound has to be delayed by that
    much before any of it can be delivered, so the stage primes with silence
    for this many samples. Measured, never assumed: it differs per rate (and
    would differ again if the converter were swapped out).
    """
    if rate == _SAMPLE_RATE:
        return 0

    down = rtc.AudioResampler(
        input_rate=rate,
        output_rate=_SAMPLE_RATE,
        num_channels=1,
        quality=rtc.AudioResamplerQuality.MEDIUM,
    )
    up = rtc.AudioResampler(
        input_rate=_SAMPLE_RATE,
        output_rate=rate,
        num_channels=1,
        quality=rtc.AudioResamplerQuality.MEDIUM,
    )
    n_native = int(rate * _FRAME_SAMPLES / _SAMPLE_RATE)
    silence = np.zeros(n_native, dtype=np.int16)
    pushed = 0
    delivered = 0
    peak = 0
    for _ in range(int(seconds * 100)):
        frame = rtc.AudioFrame(
            data=silence.tobytes(),
            sample_rate=rate,
            num_channels=1,
            samples_per_channel=n_native,
        )
        pushed += n_native
        blocks = [_frame_to_float(f) for f in down.push(frame)]
        if blocks:
            block = np.concatenate(blocks)
            block = block[: len(block) - len(block) % _FRAME_SAMPLES]
            if len(block):
                for f in up.push(_to_int16_frame(block, _SAMPLE_RATE)):
                    delivered += f.samples_per_channel
        peak = max(peak, pushed - delivered)
    return peak


class _QueueBuffer:
    """Preallocated linear FIFO buffer for 1-D float32 arrays.

    Eliminates per-frame ``np.concatenate`` allocations on the hot path.
    Compacted only when growth is necessary (amortized O(1)).
    """

    def __init__(self, initial_capacity: int = 8192):
        self._buf = np.empty(initial_capacity, dtype=np.float32)
        self._start = 0
        self._end = 0

    def __len__(self) -> int:
        return self._end - self._start

    def _compact(self) -> None:
        n = len(self)
        if n > 0:
            self._buf[:n] = self._buf[self._start : self._end]
        self._start = 0
        self._end = n

    def _ensure(self, extra: int) -> None:
        needed = self._end + extra
        if needed > len(self._buf):
            self._compact()
            needed = self._end + extra
            if needed > len(self._buf):
                new_cap = max(len(self._buf) * 2, needed)
                new_buf = np.empty(new_cap, dtype=np.float32)
                new_buf[: self._end] = self._buf[: self._end]
                self._buf = new_buf

    def append(self, arr: np.ndarray) -> None:
        n = len(arr)
        if n == 0:
            return
        self._ensure(n)
        self._buf[self._end : self._end + n] = arr
        self._end += n

    def peek(self, n: int) -> np.ndarray:
        n = min(n, len(self))
        return self._buf[self._start : self._start + n]

    def popleft(self, n: int) -> None:
        self._start += min(n, len(self))

    def clear(self) -> None:
        self._start = 0
        self._end = 0


class _StreamStage:
    """Decouples frame-in from hop-in for one stream.

    A frame arriving at the native rate is converted to 16 kHz, cut into
    whole 160-sample hops, run through the model, blended with its own dry
    signal and converted back. ``push()`` always returns exactly as many
    samples as it was given.

    A streaming resampler returns its samples in bursts, so it routinely
    has nothing to hand back, and its burst size is not a multiple of the
    hop. The output FIFO therefore absorbs the deficit: while it cannot
    fill a frame, the stage emits silence (the *prime*). Because input and
    output rates are equal in the long run, the prime settles at exactly
    the converters' backlog and no further shortfall occurs; a shortfall
    after the first successful frame is an *underrun* and is counted.
    """

    def __init__(
        self,
        strength: float,
        process_hop: HopProcessor,
        debug_logging: bool = False,
    ) -> None:
        self._strength = strength
        self._process_hop = process_hop
        self._debug_logging = debug_logging
        self._debug_frame_count = 0

        self._in_q = _QueueBuffer()  # 16 kHz, awaiting a whole hop
        self._wet_q = _QueueBuffer()  # 16 kHz, model output awaiting blend
        self._dry_q = _QueueBuffer() if strength < 1.0 else None  # 16 kHz input
        self._out_q = _QueueBuffer()  # native rate, awaiting delivery

        self._down: rtc.AudioResampler | None = None
        self._up: rtc.AudioResampler | None = None
        self._rate = 0
        self._backlog = 0

        # Counters. Never reset: they describe the lifetime of the stream.
        self.hops_processed = 0
        self.startup_fill_samples = 0  # expected: the prime
        self.underruns = 0  # unexpected: must stay 0 in steady state
        self.model_failures = 0

    @property
    def rate(self) -> int:
        """Native rate the stage is configured for, or 0 before the first frame."""
        return self._rate

    @property
    def pending_output_samples(self) -> int:
        """Samples accepted but not yet delivered (dropped when the stream ends)."""
        return len(self._out_q)

    @property
    def backlog_samples(self) -> int:
        """Native-rate samples the converters hold back at their worst."""
        return self._backlog

    def reset(self) -> None:
        """Forget everything buffered. The next frame re-derives the rate."""
        self._in_q.clear()
        self._wet_q.clear()
        self._out_q.clear()
        if self._dry_q is not None:
            self._dry_q.clear()
        self._down = None
        self._up = None
        self._rate = 0
        self._backlog = 0
        self._debug_frame_count = 0

    def rebuild(self, rate: int) -> None:
        """(Re)configure for a native sample rate, discarding buffered audio."""
        self.reset()
        self._rate = rate
        self._backlog = _measure_backlog(rate)
        if rate != _SAMPLE_RATE:
            self._down = rtc.AudioResampler(
                input_rate=rate,
                output_rate=_SAMPLE_RATE,
                num_channels=1,
                quality=rtc.AudioResamplerQuality.MEDIUM,
            )
            self._up = rtc.AudioResampler(
                input_rate=_SAMPLE_RATE,
                output_rate=rate,
                num_channels=1,
                quality=rtc.AudioResamplerQuality.MEDIUM,
            )

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Feed one frame's samples in; get exactly as many back."""
        need = len(samples)
        if need == 0:
            return np.zeros(0, dtype=np.float32)

        # 1. native rate → 16 kHz (the converter hands back 0..N samples)
        if self._down is not None:
            blocks = [
                _frame_to_float(f)
                for f in self._down.push(_to_int16_frame(samples, self._rate))
            ]
            arrived = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
        else:
            arrived = samples

        if len(arrived):
            self._in_q.append(arrived)
            if self._dry_q is not None:
                self._dry_q.append(arrived)

        # 2. whole hops → model
        while len(self._in_q) >= _FRAME_SAMPLES:
            hop = self._in_q.peek(_FRAME_SAMPLES)
            self._in_q.popleft(_FRAME_SAMPLES)
            self._wet_q.append(self._run_hop(hop))

        # 3. wet/dry in one timebase, then 16 kHz → native
        self._flush_wet()

        # 4. prime, then deliver exactly what arrived
        if self.startup_fill_samples < self._prime_target(need):
            # The converters are still filling. Nothing buffered is consumed,
            # so every sample arrives late — none is lost, none is bypassed.
            self.startup_fill_samples += need
            return np.zeros(need, dtype=np.float32)

        if len(self._out_q) < need:
            return self._on_shortfall(need)

        out = self._out_q.peek(need).copy()
        self._out_q.popleft(need)
        return out

    def _run_hop(self, hop: np.ndarray) -> np.ndarray:
        try:
            out = self._process_hop(hop)
        except Exception:
            # The host's own fallback is to hand the raw frame downstream, so
            # never let an inference failure become a leak: keep the clock
            # honest with silence and surface it in the counters.
            self.model_failures += 1
            if self.model_failures == 1:
                logger.exception(
                    "Hush inference failed; emitting silence for this hop "
                    "(failures=%d)",
                    self.model_failures,
                )
            out = np.zeros(_FRAME_SAMPLES, dtype=np.float32)

        self.hops_processed += 1
        if self._debug_logging:
            if self._debug_frame_count % 100 == 0:
                logger.debug(
                    "Hush frame: input_rms=%.5f output_rms=%.5f strength=%.2f",
                    float(np.sqrt(np.mean(hop**2))),
                    float(np.sqrt(np.mean(out**2))),
                    self._strength,
                )
            self._debug_frame_count += 1
        return out

    def _flush_wet(self) -> None:
        """Blend and convert everything the model has produced so far."""
        while len(self._wet_q) > 0:
            n = len(self._wet_q)
            wet = self._wet_q.peek(n).copy()
            self._wet_q.popleft(n)
            if self._dry_q is not None:
                # Dry is appended in lockstep with the input, so its head is
                # the same instant as the wet head — never a different one.
                dry = self._dry_q.peek(n)
                self._dry_q.popleft(n)
                wet = self._strength * wet + (1.0 - self._strength) * dry
            self._to_native(wet)

    def _to_native(self, block: np.ndarray) -> None:
        if self._up is None:
            self._out_q.append(block)
            return
        for f in self._up.push(_to_int16_frame(block, _SAMPLE_RATE)):
            self._out_q.append(_frame_to_float(f))

    def _prime_target(self, need: int) -> int:
        """Silence to emit before the enhanced path can be trusted to keep up.

        The measured worst-case converter backlog, plus a hop's worth for the
        16 kHz remainder the stage itself hides, plus one more when the native
        hop is not a whole number of frames. Deterministic: priming runs to
        this count and does not stop at the first frame that happens to fill,
        because that is exactly the situation that leaks or drops later. Zero
        at 16 kHz — no converter, so nothing to cover.
        """
        if self._backlog == 0:
            # No converter: only the 16 kHz remainder can leave us short.
            hop_native = _FRAME_SAMPLES * self._rate // _SAMPLE_RATE
            return hop_native if self._rate and need % hop_native else 0
        hop_native = _FRAME_SAMPLES * self._rate // _SAMPLE_RATE
        extra = hop_native if need % hop_native else 0
        return self._backlog + hop_native + extra

    def _on_shortfall(self, need: int) -> np.ndarray:
        self.underruns += 1
        if self.underruns == 1:
            logger.warning(
                "Hush output FIFO underrun (%d samples wanted); emitting silence",
                need,
            )
        return np.zeros(need, dtype=np.float32)


class HushNoiseSuppressor(rtc.FrameProcessor[rtc.AudioFrame]):
    """In-process Hush noise suppressor for self-hosted LiveKit.

    See ``noise_suppression()`` in ``__init__.py`` for parameter documentation.
    """

    def __init__(
        self,
        model_path: str | None = None,
        atten_lim_db: float = 100.0,
        strength: float = 0.5,
        debug_logging: bool = False,
    ) -> None:
        if atten_lim_db < 0:
            logger.warning(
                "atten_lim_db=%g is negative; clamping to 0 (no attenuation limit). "
                "Negative values boost gain instead of limiting attenuation.",
                atten_lim_db,
            )
            atten_lim_db = 0.0
        if model_path is None:
            model_path = _DEFAULT_MODEL_DIR

        # Shared ONNX model (one per process, loaded lazily on first use)
        shared_model = _get_shared_model(model_path)
        self._session: HushSession | None = HushSession(shared_model, atten_lim_db)

        self._strength = max(0.0, min(1.0, strength))
        self._debug_logging = debug_logging
        self._stage = _StreamStage(
            strength=self._strength,
            process_hop=self._process_hop,
            debug_logging=debug_logging,
        )

        self._enabled = True

    # ------------------------------------------------------------------ #
    # FrameProcessor interface                                             #
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if value and not self._enabled:
            # Resuming after a pause: stale audio in the buffers belongs to a
            # moment the consumer never heard.
            self._stage.reset()
        self._enabled = value

    @property
    def hops_processed(self) -> int:
        """Model hops run so far — must advance even when frames are large."""
        return self._stage.hops_processed

    @property
    def startup_fill_samples(self) -> int:
        """Samples of priming silence emitted at the head of the stream."""
        return self._stage.startup_fill_samples

    @property
    def underruns(self) -> int:
        """Frames that could not be filled after the prime — expected to be 0."""
        return self._stage.underruns

    @property
    def model_failures(self) -> int:
        """Hops where inference raised and silence was substituted."""
        return self._stage.model_failures

    def _on_stream_info_updated(
        self,
        *,
        room_name: str,
        participant_identity: str,
        publication_sid: str,
    ) -> None:
        self._stage.reset()
        if self._session is not None:
            self._session.reset_state()

    def _process_hop(self, hop: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("Hush session is closed")
        return self._session.process_frame(hop)

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled or self._strength <= 0.0:
            # Intentional bypass: untouched, no inference, no resampling.
            return frame

        if frame.sample_rate != self._stage.rate:
            self._stage.rebuild(frame.sample_rate)

        samples = _frame_to_float(frame)
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1)

        out_samples = self._stage.push(samples)

        if frame.num_channels > 1:
            out_samples = np.repeat(out_samples, frame.num_channels)

        out_int16 = (np.clip(out_samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        return rtc.AudioFrame(
            data=out_int16.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )

    def _close(self) -> None:
        self._enabled = False
        pending = self._stage.pending_output_samples
        if pending:
            logger.debug("closing Hush stream, discarding %d buffered samples", pending)
        self._stage.reset()
        if self._session is not None:
            self._session.close()
            self._session = None

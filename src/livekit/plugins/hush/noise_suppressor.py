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
resamples and pads with silence where the converters' declared delay leaves a
gap, so the model never has to be bypassed. See
``docs/adr/0001-no-leak-enhanced-path.md``.
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
from ._resampler import PolyphaseResampler

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

#: Converts one 160-sample hop at 16 kHz into enhanced audio.
HopProcessor = Callable[[np.ndarray], np.ndarray]


def _frame_to_float(frame: rtc.AudioFrame) -> np.ndarray:
    return np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0


def _silence_like(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    """A frame's worth of silence, shaped like `frame`."""
    pcm = np.zeros(frame.samples_per_channel * frame.num_channels, dtype=np.int16)
    return rtc.AudioFrame(
        data=pcm.tobytes(),
        sample_rate=frame.sample_rate,
        num_channels=frame.num_channels,
        samples_per_channel=frame.samples_per_channel,
    )


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

    Each converter declares a delay of half its kernel. The output FIFO
    absorbs whatever that delay, the hop buffer and the frame boundary leave
    short: a frame that cannot be filled with audio carries silence first and
    the audio that is ready after it (the *prime*), never a whole silent
    frame just because the converters are still filling. The prime therefore
    self-sizes to the declared delay plus at most one hop. A later shortfall
    larger than a hop means the enhanced path is not keeping up after all;
    that is an *underrun* and is counted.
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
        self._debug_hop_count = 0

        self._in_q = _QueueBuffer()  # 16 kHz, awaiting a whole hop
        self._wet_q = _QueueBuffer()  # 16 kHz, model output awaiting blend
        self._dry_q = _QueueBuffer() if strength < 1.0 else None  # 16 kHz input
        self._out_q = _QueueBuffer()  # native rate, awaiting delivery

        self._down: PolyphaseResampler | None = None
        self._up: PolyphaseResampler | None = None
        self._rate = 0
        self._hop_native = 0
        self._declared_delay = 0
        # The stream has delivered a full frame, or has paid the declared
        # prime; from then on a large shortfall is unexpected.
        self._primed = False

        # Counters. Never reset: they describe the lifetime of the stream.
        self.hops_processed = 0
        self.prime_samples = 0  # silence emitted to cover delay + hop jitter
        self.underruns = 0  # unexpected: must stay 0 in steady state
        self.failures = 0  # inference or processing failures, answered with silence

    @property
    def rate(self) -> int:
        """Native rate the stage is configured for, or 0 before the first frame."""
        return self._rate

    @property
    def pending_output_samples(self) -> int:
        """Samples accepted but not yet delivered (dropped when the stream ends)."""
        return len(self._out_q)

    @property
    def declared_delay_samples(self) -> int:
        """Native samples of delay the converters declare (their kernel/2)."""
        return self._declared_delay

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
        self._hop_native = 0
        self._declared_delay = 0
        self._primed = False
        self._debug_hop_count = 0

    def rebuild(self, rate: int) -> None:
        """(Re)configure for a native sample rate, discarding buffered audio."""
        self.reset()
        self._rate = rate
        self._hop_native = _FRAME_SAMPLES * rate // _SAMPLE_RATE
        if rate == _SAMPLE_RATE:
            return
        self._down = PolyphaseResampler(rate, _SAMPLE_RATE)
        self._up = PolyphaseResampler(_SAMPLE_RATE, rate)
        # Down's delay is in native input samples; the up converter's
        # withheld count is already in native output samples.
        self._declared_delay = (
            self._down.delay_input_samples + self._up.delay_output_samples
        )

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Feed one frame's samples in; get exactly as many back."""
        need = len(samples)
        if need == 0:
            return np.zeros(0, dtype=np.float32)

        # 1. native rate → 16 kHz. The converter withholds its declared
        # delay at the head; the prime below pays for it.
        arrived = self._down.push(samples) if self._down is not None else samples

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

        # 4. Deliver exactly what arrived. A shortfall is primed with silence
        # at the head of the same frame, so priming is never quantised to a
        # whole frame: the audio starts where the converters' delay ends.
        available = len(self._out_q)
        if available >= need:
            out = self._out_q.peek(need).copy()
            self._out_q.popleft(need)
            self._primed = True
            return out

        gap = need - available
        out = np.zeros(need, dtype=np.float32)
        if available:
            out[gap:] = self._out_q.peek(available)
            self._out_q.popleft(available)
        self.prime_samples += gap

        if self._primed and self._hop_native and gap > self._hop_native:
            self.underruns += 1
            if self.underruns == 1:
                logger.warning(
                    "Hush output FIFO underrun (%d samples short); emitting silence",
                    gap,
                )
        if self.prime_samples >= self._declared_delay + self._hop_native:
            self._primed = True
        return out

    def record_failure(self, what: str) -> None:
        """Count a failure of `what` and say so once; the caller emits silence.

        The host's own fallback is to hand the raw frame downstream, so a
        failure must never become a leak. It is also never silent to operators:
        the counter stays visible even though only the first one logs.
        """
        self.failures += 1
        if self.failures == 1:
            logger.exception("Hush %s failed; emitting silence", what)

    def _run_hop(self, hop: np.ndarray) -> np.ndarray:
        try:
            out = self._process_hop(hop)
        except Exception:  # noqa: BLE001 - ADR-0001: silence, never a leak
            self.record_failure("inference")
            out = np.zeros(_FRAME_SAMPLES, dtype=np.float32)

        self.hops_processed += 1
        if self._debug_logging:
            if self._debug_hop_count % 100 == 0:
                logger.debug(
                    "Hush hop: input_rms=%.5f output_rms=%.5f strength=%.2f",
                    float(np.sqrt(np.mean(hop**2))),
                    float(np.sqrt(np.mean(out**2))),
                    self._strength,
                )
            self._debug_hop_count += 1
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
        if self._up is not None:
            block = self._up.push(block)
        self._out_q.append(block)


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
            # Resuming after a pause: buffered audio and DF history both belong
            # to a moment the consumer never heard.
            self._stage.reset()
            if self._session is not None:
                self._session.reset_state()
        self._enabled = value

    @property
    def hops_processed(self) -> int:
        """Model hops run so far — must advance even when frames are large."""
        return self._stage.hops_processed

    @property
    def prime_samples(self) -> int:
        """Silence emitted to cover the converters' delay, across this instance."""
        return self._stage.prime_samples

    @property
    def declared_delay_samples(self) -> int:
        """Native samples of delay the converters declare (their kernel/2)."""
        return self._stage.declared_delay_samples

    @property
    def underruns(self) -> int:
        """Shortfalls larger than one hop after priming — expected to be 0."""
        return self._stage.underruns

    @property
    def failures(self) -> int:
        """Inference or processing failures, each answered with silence."""
        return self._stage.failures

    @property
    def pending_output_samples(self) -> int:
        """Samples sitting in the output FIFO — dropped when the stream closes."""
        return self._stage.pending_output_samples

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

        try:
            return self._enhance(frame)
        except Exception:  # noqa: BLE001 - ADR-0001: silence, never a leak
            # A raising _process is a leak by another route: the host's own
            # fallback hands the raw frame downstream. Silence keeps the clock
            # and the no-leak guarantee, and the traceback names the fault.
            self._stage.record_failure("processing")
            return _silence_like(frame)

    def _enhance(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
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

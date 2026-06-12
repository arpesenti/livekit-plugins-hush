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
"""

import logging
import os
from typing import Optional

import numpy as np
from livekit import rtc

from ._hush_model import (
    HushSession,
    _CHUNK_SAMPLES,
    _SAMPLE_RATE,
    _get_shared_model,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_DEFAULT_MODEL_FILE = "hush_dfnet_se.onnx"


class HushNoiseSuppressor(rtc.FrameProcessor[rtc.AudioFrame]):
    """In-process Hush noise suppressor for self-hosted LiveKit.

    Pass to ``AudioInputOptions(noise_cancellation=hush.noise_suppression())``.
    Each instance is independent — create one per call session. The ONNX
    model is shared across instances within the same process.

    The model processes audio in fixed 32-frame (320ms) chunks for GRU
    context. Audio is accumulated until 5120 samples (32 × 160) are
    available, then processed as a batch. First-chunk latency is 320ms.

    Parameters
    ----------
    model_path : str, optional
        Path to the exported ONNX model file.
    atten_lim_db : float
        Maximum attenuation in dB (default 100.0 = unlimited).
    strength : float
        Wet/dry blend factor. 0.0 = bypass, 1.0 = full suppression.
    debug_logging : bool
        Log per-chunk diagnostics every 10 chunks at DEBUG level.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        atten_lim_db: float = 100.0,
        strength: float = 0.5,
        debug_logging: bool = False,
    ) -> None:
        if model_path is None:
            model_path = os.path.join(_DEFAULT_MODEL_DIR, _DEFAULT_MODEL_FILE)

        # Shared ONNX model (one per process, loaded lazily on first use)
        shared_model = _get_shared_model(model_path, atten_lim_db)
        self._session = HushSession(shared_model)

        # Input/output queues for handling arbitrary LiveKit frame sizes
        self._input_queue = np.zeros(0, dtype=np.float32)
        self._output_queue = np.zeros(0, dtype=np.float32)
        self._dry_queue = np.zeros(0, dtype=np.float32)

        self._strength = max(0.0, min(1.0, strength))
        self._debug_logging = debug_logging
        self._debug_chunk_count = 0

        # Resamplers — created lazily on the first frame
        self._downsampler: rtc.AudioResampler | None = None
        self._upsampler: rtc.AudioResampler | None = None
        self._native_rate: int = 0

        self._enabled = True

    # ------------------------------------------------------------------ #
    # FrameProcessor interface                                             #
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled:
            return frame

        # Lazily create resamplers when we learn the incoming sample rate
        if frame.sample_rate != self._native_rate:
            # Flush existing resamplers before recreating them so internal
            # filter state / buffered samples are drained rather than lost.
            if self._downsampler is not None:
                flushed = self._downsampler.flush()
                for f in flushed:
                    s = np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
                    self._input_queue = np.concatenate([self._input_queue, s])
            if self._upsampler is not None:
                self._upsampler.flush()

            self._native_rate = frame.sample_rate
            if frame.sample_rate != _SAMPLE_RATE:
                self._downsampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=_SAMPLE_RATE,
                    num_channels=1,
                    quality=rtc.AudioResamplerQuality.MEDIUM,
                )
                self._upsampler = rtc.AudioResampler(
                    input_rate=_SAMPLE_RATE,
                    output_rate=frame.sample_rate,
                    num_channels=1,
                    quality=rtc.AudioResamplerQuality.MEDIUM,
                )
            else:
                self._downsampler = None
                self._upsampler = None

        # Convert int16 → float32 mono
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1)

        # Build a mono AudioFrame for the resampler
        mono_int16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        mono_frame = rtc.AudioFrame(
            data=mono_int16.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=len(mono_int16),
        )

        # Downsample to 16 kHz
        if self._downsampler is not None:
            frames_16k = self._downsampler.push(mono_frame)
        else:
            frames_16k = [mono_frame]

        if not frames_16k:
            return frame

        samples_16k = np.concatenate(
            [
                np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
                for f in frames_16k
            ]
        )

        self._input_queue = np.concatenate([self._input_queue, samples_16k])
        if self._strength < 1.0:
            self._dry_queue = np.concatenate([self._dry_queue, samples_16k])

        # Process in 32-frame chunks (5120 samples)
        while len(self._input_queue) >= _CHUNK_SAMPLES:
            chunk_in = self._input_queue[:_CHUNK_SAMPLES]
            self._input_queue = self._input_queue[_CHUNK_SAMPLES:]

            chunk_out = self._session.process_chunk(chunk_in)

            self._output_queue = np.concatenate([self._output_queue, chunk_out])

            if self._debug_logging and self._debug_chunk_count % 10 == 0:
                logger.debug(
                    "Hush chunk: input_rms=%.5f output_rms=%.5f strength=%.2f",
                    float(np.sqrt(np.mean(chunk_in**2))),
                    float(np.sqrt(np.mean(chunk_out**2))),
                    self._strength,
                )
            self._debug_chunk_count += 1

        # Drain the same number of samples that went IN
        n_16k = len(samples_16k)
        if len(self._output_queue) < n_16k:
            return frame  # filling up during startup latency

        out_16k = self._output_queue[:n_16k]
        self._output_queue = self._output_queue[n_16k:]

        # Wet/dry blend
        if self._strength < 1.0:
            dry_16k = self._dry_queue[:n_16k]
            self._dry_queue = self._dry_queue[n_16k:]
            out_16k = self._strength * out_16k + (1.0 - self._strength) * dry_16k

        # Build 16 kHz AudioFrame and upsample back
        out_int16_16k = (np.clip(out_16k, -1.0, 1.0) * 32767.0).astype(np.int16)
        out_frame_16k = rtc.AudioFrame(
            data=out_int16_16k.tobytes(),
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=len(out_int16_16k),
        )

        if self._upsampler is not None:
            out_frames = self._upsampler.push(out_frame_16k)
        else:
            out_frames = [out_frame_16k]

        if not out_frames:
            return frame

        out_samples = np.concatenate(
            [
                np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
                for f in out_frames
            ]
        )

        # Trim or pad to exactly match the input frame length
        target = frame.samples_per_channel
        if len(out_samples) > target:
            out_samples = out_samples[:target]
        elif len(out_samples) < target:
            out_samples = np.pad(out_samples, (0, target - len(out_samples)))

        # Restore original channel count
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
        if hasattr(self, "_session"):
            self._session.close()
            self._session = None  # type: ignore[assignment]

"""Tests for Hush noise suppression plugin.

Unit tests mock the ONNX model. Integration tests use the committed real model;
locally they skip if it is unavailable, but CI requires them to run.

Run: uv run pytest
"""

import os

import numpy as np
import pytest

from livekit import rtc

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def create_audio_frame(
    samples: np.ndarray,
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> rtc.AudioFrame:
    """Create a LiveKit AudioFrame from float32 samples in [-1, 1]."""
    assert samples.dtype == np.float32
    int16_samples = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    return rtc.AudioFrame(
        data=int16_samples.tobytes(),
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=len(int16_samples) // num_channels,
    )


def frame_to_float(frame: rtc.AudioFrame) -> np.ndarray:
    """AudioFrame back to float32 samples in [-1, 1]."""
    return np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0


# ------------------------------------------------------------------ #
# Mock HushModel                                                        #
# ------------------------------------------------------------------ #


class MockHushSession:
    """Mock session that applies a simple gain reduction per frame."""

    def __init__(self, model=None, atten_lim_db=None):
        self.gain = 0.5
        self.frames_processed = 0
        self._h_enc = None
        self._h_erb_dec = None
        self._h_df_dec = None
        self._prev_df = None

    def process_frame(self, audio: np.ndarray) -> np.ndarray:
        self.frames_processed += 1
        return audio * self.gain

    def reset_state(self):
        self.frames_processed = 0

    def close(self):
        pass


class MockHushModel:
    """Mock shared model."""

    def __init__(self, model_path=None):
        self.session = object()


def _patch_module(monkeypatch):
    """Patch the noise_suppressor module with mocks."""
    import livekit.plugins.hush.noise_suppressor as ns_module

    monkeypatch.setattr(ns_module, "HushSession", MockHushSession)
    monkeypatch.setattr(ns_module, "_FRAME_SAMPLES", 160)
    mock_model = MockHushModel()

    def mock_get_shared(*args, **kwargs):
        return mock_model

    monkeypatch.setattr(ns_module, "_get_shared_model", mock_get_shared)


@pytest.fixture
def mock_suppressor(monkeypatch):
    """Create a HushNoiseSuppressor with mocked model and session."""
    _patch_module(monkeypatch)
    import livekit.plugins.hush.noise_suppressor as ns_module

    return ns_module.HushNoiseSuppressor(strength=1.0)


# ------------------------------------------------------------------ #
# Frame processing tests                                                #
# ------------------------------------------------------------------ #


class TestFrameProcessing:
    """Tests for HushNoiseSuppressor._process with mocked inference."""

    def test_shorter_than_a_hop_is_not_passed_through(self, mock_suppressor):
        """A frame shorter than one hop is primed with silence, never echoed back."""
        rng = np.random.default_rng(42)
        first = rng.uniform(-0.5, 0.5, 80).astype(np.float32)
        result = mock_suppressor._process(create_audio_frame(first))
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == 80
        assert np.all(out == 0.0), "the prime must be silence, not the input"
        assert mock_suppressor.hops_processed == 0

        # One hop of lead covers the hop buffer; priming is sub-frame, so the
        # second frame is already audio for the first frame's input.
        second = rng.uniform(-0.5, 0.5, 80).astype(np.float32)
        result = mock_suppressor._process(create_audio_frame(second))
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert mock_suppressor.prime_samples == 80
        assert mock_suppressor.hops_processed == 1
        np.testing.assert_array_almost_equal(out, first * 0.5, decimal=3)

        third = rng.uniform(-0.5, 0.5, 80).astype(np.float32)
        result = mock_suppressor._process(create_audio_frame(third))
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == 80
        assert mock_suppressor.hops_processed == 1
        assert mock_suppressor.underruns == 0
        np.testing.assert_array_almost_equal(out, second * 0.5, decimal=3)

    def test_exact_frame(self, mock_suppressor):
        """Exactly one 10 ms frame (160 samples)."""
        samples = np.random.default_rng(42).uniform(-0.5, 0.5, 160).astype(np.float32)
        frame = create_audio_frame(samples)
        result = mock_suppressor._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == 160
        reduction_ratio = np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(samples**2))
        assert 0.4 < reduction_ratio < 0.6

    def test_multiple_frames(self, mock_suppressor):
        """Multiple frames in one input."""
        samples = (
            np.random.default_rng(42).uniform(-0.5, 0.5, 5 * 160).astype(np.float32)
        )
        frame = create_audio_frame(samples)
        result = mock_suppressor._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == 5 * 160

    def test_non_multiple_frame_size(self, mock_suppressor):
        """Non-aligned frame sizes should not produce silence gaps."""
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.5, 0.5, 200).astype(np.float32)
        frame = create_audio_frame(samples)
        result = mock_suppressor._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == 200

    def test_continuity_across_frames(self, mock_suppressor):
        """Processing consecutive frames produces continuous output."""
        rng = np.random.default_rng(42)
        n_frames = 10
        outputs = []

        for _ in range(n_frames):
            samples = rng.uniform(-0.5, 0.5, 160).astype(np.float32)
            frame = create_audio_frame(samples)
            result = mock_suppressor._process(frame)
            out = (
                np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            outputs.append(out)

        for i, out in enumerate(outputs):
            assert len(out) == 160, f"Frame {i} output length mismatch"
            assert np.sqrt(np.mean(out**2)) > 1e-6, f"Frame {i} is silent"

    def test_disabled_passthrough(self, mock_suppressor):
        """When disabled, output should be identical to input."""
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.5, 0.5, 160).astype(np.float32)
        frame = create_audio_frame(samples)
        mock_suppressor.enabled = False
        result = mock_suppressor._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(out, samples, decimal=4)

    def test_strength_blend(self, monkeypatch):
        """Strength=0 → passthrough, strength=1 → full suppression."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.5, 0.5, 160).astype(np.float32)

        # strength=0 (bypass)
        ns_bypass = ns_module.HushNoiseSuppressor(strength=0.0)
        frame = create_audio_frame(samples)
        result = ns_bypass._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(out, samples, decimal=4)

        # strength=1 (full suppression)
        ns_full = ns_module.HushNoiseSuppressor(strength=1.0)
        frame = create_audio_frame(samples)
        result = ns_full._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        ratio = np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(samples**2))
        assert 0.4 < ratio < 0.6

    def test_channel_restoration(self, mock_suppressor):
        """Stereo input should produce stereo output."""
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.5, 0.5, 160 * 2).astype(np.float32)
        int16_samples = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = rtc.AudioFrame(
            data=int16_samples.tobytes(),
            sample_rate=16000,
            num_channels=2,
            samples_per_channel=160,
        )
        result = mock_suppressor._process(frame)
        assert result.num_channels == 2
        assert result.samples_per_channel == 160

    @pytest.mark.parametrize("rate", (8000, 48000))
    @pytest.mark.parametrize("mode", ("strength", "disabled"))
    def test_bypass_is_free(self, monkeypatch, rate, mode):
        """A bypass returns the frame itself: no inference, no resampling."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        if mode == "strength":
            ns = ns_module.HushNoiseSuppressor(strength=0.0)
        else:
            ns = ns_module.HushNoiseSuppressor(strength=1.0)
            ns.enabled = False

        samples = (
            np.random.default_rng(1)
            .uniform(-0.5, 0.5, int(rate * 0.010))
            .astype(np.float32)
        )
        frame = create_audio_frame(samples, sample_rate=rate)
        result = ns._process(frame)

        assert result is frame
        assert ns.hops_processed == 0
        assert ns.prime_samples == 0
        assert ns._stage.rate == 0, "a bypass must not even build a converter"

    def test_reenabling_does_not_replay_stale_audio(self, monkeypatch):
        """Audio buffered before a pause belongs to a moment nobody heard."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(4)

        def frame(value=None):
            samples = (
                np.full(80, value, dtype=np.float32)
                if value is not None
                else rng.uniform(-0.5, 0.5, 80).astype(np.float32)
            )
            return create_audio_frame(samples, sample_rate=8000)

        for _ in range(40):  # fill the converters and the output FIFO
            ns._process(frame())
        assert ns.hops_processed > 0

        ns.enabled = False
        for _ in range(5):
            ns._process(frame())
        ns.enabled = True

        result = ns._process(frame(0.4))
        out = frame_to_float(result)
        assert np.all(out == 0.0), (
            "re-enabling must re-prime, not replay buffered audio"
        )

        # ...and it must actually resume, not stay silent forever.
        audible = None
        for _ in range(40):
            out = frame_to_float(ns._process(frame(0.4)))
            if np.any(out != 0.0):
                audible = out
                break
        assert audible is not None, "the stream never resumed after re-enabling"
        # The converters ramp for a sample or two, so judge the settled level.
        assert np.isclose(np.median(audible), 0.2, atol=2e-2), (
            f"resumed audio should be the model's 0.5x of 0.4, "
            f"got {np.median(audible):.4f}"
        )
        assert ns.underruns == 0

    def test_declared_delay_is_exposed_from_the_stage(self, mock_suppressor):
        """The diagnostic reports the converter delay, not the measured prime."""
        assert mock_suppressor.declared_delay_samples == 0
        frame = create_audio_frame(np.zeros(480, dtype=np.float32), sample_rate=48000)
        mock_suppressor._process(frame)
        assert mock_suppressor.declared_delay_samples > 0

    def test_processing_failure_is_counted_and_silent(self, monkeypatch):
        """A failure outside a hop must not raise, and must not leak the input."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        monkeypatch.setattr(
            ns._stage,
            "rebuild",
            lambda rate: (_ for _ in ()).throw(RuntimeError("nope")),
        )

        samples = np.full(160, 0.3, dtype=np.float32)
        result = ns._process(create_audio_frame(samples))

        out = frame_to_float(result)
        assert np.all(out == 0.0), "a failure must emit silence, never the input"
        assert ns.failures == 1
        assert ns.underruns == 0

    def test_close_discards_buffered_audio(self, monkeypatch, caplog):
        """The tail has nowhere to go at close; it is dropped, reported and freed."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(5)
        for _ in range(60):
            ns._process(
                create_audio_frame(
                    rng.uniform(-0.5, 0.5, 100).astype(np.float32), sample_rate=8000
                )
            )
        pending = ns.pending_output_samples
        assert pending > 0

        caplog.set_level("DEBUG", logger="livekit.plugins.hush.noise_suppressor")
        ns._close()

        assert "discarding" in caplog.text
        assert str(pending) in caplog.text, "the log should name the real tail"
        assert ns.pending_output_samples == 0
        assert ns._stage.rate == 0

    def test_stream_tail_is_bounded_by_the_prime(self, monkeypatch):
        """Everything accepted is delivered, bar at most one prime's worth of tail."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        outputs = []
        for _ in range(400):
            frame = create_audio_frame(
                np.full(80, 0.4, dtype=np.float32), sample_rate=8000
            )
            outputs.append(frame_to_float(ns._process(frame)))

        prime = ns.prime_samples
        assert prime > 0
        # What is still inside the stage when the stream ends is bounded by the
        # prime: the FIFO holds a burst, the converters hold the rest.
        assert ns.pending_output_samples <= prime

        # Nothing was dropped mid-stream: after priming, every frame carries the
        # model's 0.5x of the 0.4 input.
        gains = [float(np.median(o)) for o in outputs if np.any(o != 0.0)]
        assert len(gains) >= 400 - prime // 80 - 2, "frames went missing"
        assert all(abs(gain - 0.2) < 1e-2 for gain in gains), gains[:5]

    @pytest.mark.parametrize("rate", (8000, 16000, 24000, 32000, 44100, 48000))
    @pytest.mark.parametrize("frame_ms", (10, 20, 50, 100))
    def test_no_frame_is_returned_unenhanced(self, mock_suppressor, rate, frame_ms):
        """A frame never comes back as its own input, at any rate or frame size."""
        n = int(rate * frame_ms / 1000)
        rng = np.random.default_rng(0)
        for _ in range(100):
            samples = rng.uniform(-0.5, 0.5, n).astype(np.float32)
            frame = create_audio_frame(samples, sample_rate=rate)
            result = mock_suppressor._process(frame)
            quantized = (
                np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            out = (
                np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            assert result is not frame
            assert result.sample_rate == rate
            assert result.samples_per_channel == n
            assert not np.array_equal(out, quantized), (
                f"raw input leaked at {rate} Hz / {frame_ms} ms"
            )

        assert mock_suppressor.underruns == 0, (
            f"steady-state underrun at {rate} Hz / {frame_ms} ms"
        )
        assert mock_suppressor.hops_processed > 0


# ------------------------------------------------------------------ #
# Integration tests (require real model)                                #
# ------------------------------------------------------------------ #


def _model_available() -> bool:
    try:
        from livekit.plugins.hush._hush_model import HushModel

        HushModel()
        return True
    except Exception:  # noqa: BLE001 - any failure means the model is unavailable
        return False


_MODEL_AVAILABLE = _model_available()
# The weights are committed, so CI must fail rather than skip if they cannot load.
_MODEL_REQUIRED = os.getenv("CI", "").lower() == "true"
_skip_without_model = pytest.mark.skipif(
    not _MODEL_AVAILABLE and not _MODEL_REQUIRED,
    reason="ONNX model not available (skipped locally; required on CI)",
)


@_skip_without_model
class TestIntegration:
    """Integration tests with the real Hush model."""

    def test_noise_reduction(self):
        """Verify noise reduction on a synthetic noisy signal."""
        from livekit.plugins.hush import HushNoiseSuppressor

        rng = np.random.default_rng(42)
        sr = 16000
        duration = sr * 2
        t = np.arange(duration)
        signal = 0.3 * np.sin(2 * np.pi * 440.0 * t / sr)
        noise = rng.normal(0, 0.1, duration)
        noisy = (signal + noise).astype(np.float32)

        ns = HushNoiseSuppressor(strength=1.0)
        # Feed in 10 ms frames
        frame_size = 160
        outputs = []

        for i in range(0, duration - frame_size + 1, frame_size):
            frame_data = noisy[i : i + frame_size]
            frame = create_audio_frame(frame_data)
            result = ns._process(frame)
            out = (
                np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            outputs.append(out)

        output = np.concatenate(outputs)
        in_rms = np.sqrt(np.mean(noisy[: len(output)] ** 2))
        out_rms = np.sqrt(np.mean(output**2))
        assert out_rms < in_rms, (
            f"Expected noise reduction, out_rms={out_rms:.3f} >= in_rms={in_rms:.3f}"
        )

    def test_frame_160_samples(self):
        """Model processes one 10 ms frame (160 samples @ 16 kHz)."""
        from livekit.plugins.hush._hush_model import HushModel, HushSession

        model = HushModel()
        session = HushSession(model)
        rng = np.random.default_rng(42)

        result = session.process_frame(rng.uniform(-0.5, 0.5, 160).astype(np.float32))
        assert len(result) == 160
        assert result.dtype == np.float32

    def test_process_frame_after_close_raises(self):
        """A closed session refuses further frames."""
        from livekit.plugins.hush._hush_model import HushModel, HushSession

        session = HushSession(HushModel())
        session.close()
        with pytest.raises(RuntimeError, match="HushSession is closed"):
            session.process_frame(np.zeros(160, dtype=np.float32))

    def test_end_to_end_process(self):
        """End-to-end HushNoiseSuppressor._process with real model."""
        from livekit.plugins.hush import HushNoiseSuppressor

        ns = HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.5, 0.5, 5120).astype(np.float32)
        frame = create_audio_frame(samples)
        result = ns._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == 5120
        assert result.sample_rate == 16000
        assert result.num_channels == 1

    def test_stereo_with_real_model(self):
        """Stereo input through _process with real model."""
        from livekit.plugins.hush import HushNoiseSuppressor

        ns = HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.5, 0.5, 5120 * 2).astype(np.float32)
        int16_data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = rtc.AudioFrame(
            data=int16_data.tobytes(),
            sample_rate=16000,
            num_channels=2,
            samples_per_channel=5120,
        )
        result = ns._process(frame)
        assert result.num_channels == 2
        assert result.samples_per_channel == 5120
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == 5120 * 2

    def test_resampling_with_real_model(self):
        """48 kHz input through _process with real model."""
        from livekit.plugins.hush import HushNoiseSuppressor

        ns = HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.5, 0.5, 15360).astype(np.float32)
        int16_data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = rtc.AudioFrame(
            data=int16_data.tobytes(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=15360,
        )
        result = ns._process(frame)
        assert result.sample_rate == 48000
        assert result.samples_per_channel == 15360
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == 15360

    def test_multi_frame_continuity_with_real_model(self):
        """Consecutive frames produce continuous non-silent output."""
        from livekit.plugins.hush import HushNoiseSuppressor

        ns = HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)
        outputs = []
        for _ in range(5):
            samples = rng.uniform(-0.5, 0.5, 160).astype(np.float32)
            frame = create_audio_frame(samples)
            result = ns._process(frame)
            out = (
                np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            outputs.append(out)

        for i, out in enumerate(outputs):
            assert len(out) == 160, f"Frame {i} output length mismatch"
            assert np.sqrt(np.mean(out**2)) > 1e-6, f"Frame {i} is silent"

    def test_state_persists_across_frames(self):
        """Second frame uses GRU hidden state from first (state continuity)."""
        from livekit.plugins.hush._hush_model import HushModel, HushSession

        model = HushModel()
        session = HushSession(model)
        rng = np.random.default_rng(42)

        # First call: zero hidden state
        frame1 = rng.uniform(-0.5, 0.5, 160).astype(np.float32)
        out1 = session.process_frame(frame1)
        assert len(out1) == 160
        assert session._h_enc is not None
        # GRU state should now be non-zero
        assert np.abs(session._h_enc).max() > 0

        # Second call: state carried over
        frame2 = rng.uniform(-0.5, 0.5, 160).astype(np.float32)
        out2 = session.process_frame(frame2)
        assert len(out2) == 160

        # Reset clears state
        session.reset_state()
        assert session._h_enc is None or np.abs(session._h_enc).max() == 0

    def test_atten_lim_db_zero_passes_audio_through(self):
        """atten_lim_db=0 removes nothing: the output tracks the input.

        This is why "the output must differ from the input" cannot be the no-leak
        invariant — object identity plus the hop counter is.
        """
        from livekit.plugins.hush import HushNoiseSuppressor

        rng = np.random.default_rng(7)
        ns = HushNoiseSuppressor(strength=1.0, atten_lim_db=0.0)
        source, enhanced = [], []
        for _ in range(50):
            samples = rng.uniform(-0.4, 0.4, 160).astype(np.float32)
            frame = create_audio_frame(samples)
            result = ns._process(frame)
            assert result is not frame, "still a fresh frame, even at 0 dB"
            source.append(samples)
            enhanced.append(frame_to_float(result))
        assert ns.hops_processed == 50
        ns._close()

        src = np.concatenate(source)[320:]
        out = np.concatenate(enhanced)[320:]

        # The model is causal with one hop of STFT lookahead, so the passthrough
        # comes out 160 samples late: correlate over lags, don't assume position.
        lag, corr = max(
            (
                (lag, float(np.corrcoef(src[: len(src) - lag], out[lag:])[0, 1]))
                for lag in range(4 * 160)
            ),
            key=lambda pair: pair[1],
        )
        assert 0 <= lag <= 160, f"passthrough lag looks wrong: {lag} samples"
        aligned = out[lag:]
        ratio = np.sqrt(np.mean(aligned**2)) / np.sqrt(
            np.mean(src[: len(aligned)] ** 2)
        )
        assert corr > 0.999, f"atten_lim_db=0 must pass audio through: corr={corr:.4f}"
        assert 0.99 < ratio < 1.01, f"atten_lim_db=0 must not change level: {ratio:.4f}"

    def test_atten_lim_db_effect(self):
        """atten_lim_db < 100 dB limits suppression compared to unlimited."""
        from livekit.plugins.hush import HushNoiseSuppressor

        rng = np.random.default_rng(42)
        # Use real-ish audio: a mix of sine wave and noise
        sr = 16000
        t = np.arange(sr)
        speech_like = 0.3 * np.sin(2 * np.pi * 440.0 * t / sr).astype(
            np.float32
        ) + rng.normal(0, 0.1, sr).astype(np.float32)
        frame = create_audio_frame(speech_like)

        ns_unlimited = HushNoiseSuppressor(strength=1.0, atten_lim_db=100.0)
        result_u = ns_unlimited._process(frame)
        out_u = (
            np.frombuffer(result_u.data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        ns_unlimited._close()

        ns_limited = HushNoiseSuppressor(strength=1.0, atten_lim_db=6.0)
        result_l = ns_limited._process(frame)
        out_l = (
            np.frombuffer(result_l.data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        ns_limited._close()

        rms_u = np.sqrt(np.mean(out_u**2))
        rms_l = np.sqrt(np.mean(out_l**2))
        assert rms_l > rms_u, (
            f"Limited (6 dB) output RMS {rms_l:.5f} should exceed "
            f"unlimited RMS {rms_u:.5f}"
        )


# ------------------------------------------------------------------ #
# Coverage gap tests                                                    #
# ------------------------------------------------------------------ #


class TestCoverageGaps:
    """Tests specifically targeting uncovered code paths."""

    def test_resampling_path(self, monkeypatch):
        """Non-16kHz input triggers resampler creation and up/downsampling."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)

        # 48 kHz stereo input (triggers resampler + channel conversion)
        rng = np.random.default_rng(42)
        chunk_48k = rng.uniform(-0.5, 0.5, 15360 * 2).astype(np.float32)
        int16_data = (np.clip(chunk_48k, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = rtc.AudioFrame(
            data=int16_data.tobytes(),
            sample_rate=48000,
            num_channels=2,
            samples_per_channel=15360,
        )
        result = ns._process(frame)
        assert result.sample_rate == 48000
        assert result.num_channels == 2

    def test_strength_full_suppression(self, monkeypatch):
        """strength=1.0 takes the no-blend code path."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.5, 0.5, 160).astype(np.float32)
        frame = create_audio_frame(samples)
        result = ns._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        ratio = np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(samples**2))
        assert ratio < 0.8  # real suppression happened (no blend with dry)

    def test_debug_logging(self, monkeypatch):
        """debug_logging=True triggers the debug log path."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(debug_logging=True)
        rng = np.random.default_rng(42)
        for _ in range(101):  # process 101 frames to hit the log every 100
            samples = rng.uniform(-0.5, 0.5, 160).astype(np.float32)
            frame = create_audio_frame(samples)
            ns._process(frame)
        # No assertion needed — just exercises the debug log path

    def test_close(self, monkeypatch):
        """_close() cleans up the session."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor()
        assert ns.enabled is True
        ns._close()
        assert ns.enabled is False

    def test_short_audio_buffers_during_startup(self, monkeypatch):
        """16 kHz hop-aligned frames need no prime: enhanced from the first frame."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        monkeypatch.setattr(ns_module, "_FRAME_SAMPLES", 160)

        rng = np.random.default_rng(42)
        for _ in range(3):
            samples = rng.uniform(-0.5, 0.5, 160).astype(np.float32)
            result = ns._process(create_audio_frame(samples))
            out = (
                np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            assert len(out) == 160
            np.testing.assert_array_almost_equal(out, samples * 0.5, decimal=3)

        assert ns.prime_samples == 0
        assert ns.hops_processed == 3
        assert ns.underruns == 0

    def test_model_not_found_raises(self):
        """Nonexistent model path raises FileNotFoundError."""
        from livekit.plugins.hush._hush_model import HushModel

        with pytest.raises(FileNotFoundError):
            HushModel(model_path="/nonexistent/path.onnx")

    @_skip_without_model
    def test_process_frame_2d_input(self):
        """Model handles 2D array input correctly."""
        from livekit.plugins.hush._hush_model import HushModel, HushSession

        model = HushModel()
        # Two separate sessions to ensure identical state.
        s1 = HushSession(model)
        s2 = HushSession(model)
        rng = np.random.default_rng(99)
        audio_1d = rng.uniform(-0.5, 0.5, 160).astype(np.float32)
        audio_2d = audio_1d[np.newaxis, :]

        out_1d = s1.process_frame(audio_1d)
        out_2d = s2.process_frame(audio_2d)
        assert out_1d.shape == (160,)
        assert out_2d.shape == (1, 160)
        np.testing.assert_array_almost_equal(out_1d, out_2d[0], decimal=4)

    @_skip_without_model
    def test_process_frame_wrong_size(self):
        """Audio with wrong frame size raises ValueError."""
        from livekit.plugins.hush._hush_model import HushModel, HushSession

        model = HushModel()
        session = HushSession(model)
        rng = np.random.default_rng(42)
        # 80 samples — wrong (should be 160)
        short = rng.uniform(-0.5, 0.5, 80).astype(np.float32)
        with pytest.raises(ValueError, match="requires 160 samples"):
            session.process_frame(short)

    def test_output_length_matches_input_at_48k(self, monkeypatch):
        """48 kHz input: output keeps the input's length exactly."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)

        samples = rng.uniform(-0.5, 0.5, 16000).astype(np.float32)
        int16_data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = rtc.AudioFrame(
            data=int16_data.tobytes(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=16000,
        )
        result = ns._process(frame)
        assert result.samples_per_channel == 16000

    def test_output_length_matches_input_at_44k(self, monkeypatch):
        """44.1 kHz: non-integer ratio still keeps the input's length exactly."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)
        # 44.1 kHz → 16 kHz is a non-integer ratio → resampler length varies
        samples = rng.uniform(-0.5, 0.5, 4410).astype(np.float32)
        int16_data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = rtc.AudioFrame(
            data=int16_data.tobytes(),
            sample_rate=44100,
            num_channels=1,
            samples_per_channel=4410,
        )
        result = ns._process(frame)
        assert result.samples_per_channel == 4410, (
            f"Expected 4410 samples, got {result.samples_per_channel}"
        )

    def test_rate_change_rebuilds_the_converters(self, monkeypatch):
        """Mid-session rate switch rebuilds the converter state and re-primes."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)

        # First frame at 48 kHz — creates resamplers
        samples_48k = rng.uniform(-0.5, 0.5, 15360).astype(np.float32)
        int16_data = (np.clip(samples_48k, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame_48k = rtc.AudioFrame(
            data=int16_data.tobytes(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=15360,
        )
        ns._process(frame_48k)

        # Second frame at 16 kHz — flushes existing resamplers
        samples_16k = rng.uniform(-0.5, 0.5, 160).astype(np.float32)
        frame_16k = create_audio_frame(samples_16k, sample_rate=16000)
        result = ns._process(frame_16k)
        assert result.sample_rate == 16000
        assert result.samples_per_channel == 160

    def test_peak_normalization_clipping(self, monkeypatch):
        """Model output > 1.0 is clipped to [-1.0, 1.0] before int16 conversion."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        class ClippingMockSession:
            def __init__(self, model=None, atten_lim_db=None):
                pass

            def process_frame(self, audio: np.ndarray) -> np.ndarray:
                return audio * 2.0

            def reset_state(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(ns_module, "HushSession", ClippingMockSession)

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.3, 0.3, 160).astype(np.float32)
        frame = create_audio_frame(samples)
        result = ns._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert np.all(np.abs(out) <= 1.0 + 1e-6), "Output should be clipped to [-1, 1]"

    def test_stereo_different_channels(self, monkeypatch):
        """Stereo with different L/R content is collapsed to mono via mean."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)
        n_per_ch = 160
        left = rng.uniform(-0.5, 0.5, n_per_ch).astype(np.float32)
        right = rng.uniform(-0.5, 0.5, n_per_ch).astype(np.float32)
        interleaved = np.empty(n_per_ch * 2, dtype=np.float32)
        interleaved[0::2] = left
        interleaved[1::2] = right
        int16_data = (np.clip(interleaved, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = rtc.AudioFrame(
            data=int16_data.tobytes(),
            sample_rate=16000,
            num_channels=2,
            samples_per_channel=n_per_ch,
        )
        result = ns._process(frame)
        assert result.num_channels == 2
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        out = out.reshape(-1, 2)
        # Both channels should be identical (mono processing + stereo repeat)
        np.testing.assert_array_almost_equal(out[:, 0], out[:, 1], decimal=4)

    def test_on_stream_info_updated(self, monkeypatch):
        """_on_stream_info_updated resets session state."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        # Process a frame to mark state as having been used
        samples = np.random.default_rng(42).uniform(-0.5, 0.5, 160).astype(np.float32)
        ns._process(create_audio_frame(samples))
        assert ns._session.frames_processed == 1

        ns._on_stream_info_updated(
            room_name="test_room",
            participant_identity="test_participant",
            publication_sid="test_sid",
        )

        assert ns._session.frames_processed == 0

    @_skip_without_model
    def test_clean_speech_preservation(self):
        """Clean speech should pass through with minimal distortion."""
        import wave

        from livekit.plugins.hush._hush_model import HushModel, HushSession

        # Load clean speech sample
        speech_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "audio", "originals", "speech.wav"
        )
        with wave.open(speech_path, "rb") as wf:
            n_frames = wf.getnframes()
            speech = (
                np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(
                    np.float32
                )
                / 32768.0
            )

        # Process in streaming mode (per-frame)
        model = HushModel()
        session = HushSession(model)
        frame_samples = 160
        output = np.empty(len(speech), dtype=np.float32)
        pos = 0
        while pos < len(speech):
            end = min(pos + frame_samples, len(speech))
            chunk = speech[pos:end]
            if len(chunk) < frame_samples:
                chunk = np.pad(chunk, (0, frame_samples - len(chunk)))
            denoised = session.process_frame(chunk)
            n_out = min(len(denoised), end - pos)
            output[pos : pos + n_out] = denoised[:n_out]
            pos += frame_samples
        session.close()

        in_rms = np.sqrt(np.mean(speech**2))
        out_rms = np.sqrt(np.mean(output**2))
        out_peak = np.max(np.abs(output))

        # Output should retain reasonable energy
        assert out_rms > in_rms * 0.3, (
            f"Output too quiet: out_rms={out_rms:.4f} in_rms={in_rms:.4f} "
            f"ratio={out_rms / in_rms:.3f}"
        )

        # No clipping artifacts
        assert out_peak < 1.5, f"Output peak too high (clipping): {out_peak:.4f}"

        # Positive correlation with input in speech regions
        chunk = 16000  # 1-second segments
        corrs = []
        for i in range(0, len(speech) - chunk, chunk):
            seg_in = speech[i : i + chunk]
            seg_out = output[i : i + chunk]
            if np.sqrt(np.mean(seg_in**2)) > 0.01:
                corrs.append(np.corrcoef(seg_in, seg_out)[0, 1])

        avg_corr = np.mean(corrs)
        assert avg_corr > 0.0, f"Anti-correlated with input: avg_corr={avg_corr:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

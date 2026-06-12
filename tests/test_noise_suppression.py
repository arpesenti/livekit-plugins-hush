"""Tests for Hush noise suppression plugin.

Unit tests mock the ONNX model. Integration tests require the real model.

Run: python -m pytest tests/ -v
"""

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


# ------------------------------------------------------------------ #
# Mock HushModel                                                        #
# ------------------------------------------------------------------ #


class MockHushSession:
    """Mock session that applies a simple gain reduction."""

    def __init__(self, model=None):
        self.gain = 0.5
        self.processed_chunks = []

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        self.processed_chunks.append(audio.copy())
        return audio * self.gain

    def close(self):
        pass


class MockHushModel:
    """Mock shared model."""

    def __init__(self, model_path=None, atten_lim_db=100.0):
        self.session = object()


def _patch_module(monkeypatch):
    """Patch the noise_suppressor module with mocks."""
    import livekit.plugins.hush.noise_suppressor as ns_module

    monkeypatch.setattr(ns_module, "HushSession", MockHushSession)
    monkeypatch.setattr(ns_module, "_CHUNK_SAMPLES", 640)
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

    def test_smaller_than_chunk(self, mock_suppressor):
        """Fewer samples than chunk size → pass through unprocessed."""
        samples = np.random.default_rng(42).uniform(-0.5, 0.5, 80).astype(np.float32)
        frame = create_audio_frame(samples)
        result = mock_suppressor._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == 80
        np.testing.assert_array_almost_equal(out, samples, decimal=4)

    def test_exact_chunk(self, mock_suppressor):
        """Exactly one chunk (4 frames = 640 samples)."""
        chunk_samples = 4 * 160  # 640
        samples = (
            np.random.default_rng(42)
            .uniform(-0.5, 0.5, chunk_samples)
            .astype(np.float32)
        )
        frame = create_audio_frame(samples)
        result = mock_suppressor._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == chunk_samples
        reduction_ratio = np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(samples**2))
        assert 0.4 < reduction_ratio < 0.6

    def test_multiple_chunks(self, mock_suppressor):
        """Two full chunks (8 frames = 1280 samples)."""
        chunk_samples = 8 * 160
        samples = (
            np.random.default_rng(42)
            .uniform(-0.5, 0.5, chunk_samples)
            .astype(np.float32)
        )
        frame = create_audio_frame(samples)
        result = mock_suppressor._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == chunk_samples

    def test_non_multiple_chunk_size(self, mock_suppressor):
        """Non-aligned frame sizes should not produce silence gaps."""
        rng = np.random.default_rng(42)
        chunk_size = 200
        samples = rng.uniform(-0.5, 0.5, chunk_size).astype(np.float32)
        frame = create_audio_frame(samples)
        result = mock_suppressor._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == chunk_size

    def test_continuity_across_frames(self, mock_suppressor):
        """Processing consecutive frames should produce continuous output."""
        rng = np.random.default_rng(42)
        n_frames = 10
        chunk_per_frame = 4 * 160  # exactly one chunk per frame
        outputs = []

        for _ in range(n_frames):
            samples = rng.uniform(-0.5, 0.5, chunk_per_frame).astype(np.float32)
            frame = create_audio_frame(samples)
            result = mock_suppressor._process(frame)
            out = (
                np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            outputs.append(out)

        for i, out in enumerate(outputs):
            assert len(out) == chunk_per_frame, f"Frame {i} output length mismatch"
            assert np.sqrt(np.mean(out**2)) > 1e-6, f"Frame {i} is silent"

    def test_disabled_passthrough(self, mock_suppressor):
        """When disabled, output should be identical to input."""
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.5, 0.5, 640).astype(np.float32)
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
        samples = rng.uniform(-0.5, 0.5, 640).astype(np.float32)

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
        samples = rng.uniform(-0.5, 0.5, 640 * 2).astype(np.float32)
        int16_samples = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = rtc.AudioFrame(
            data=int16_samples.tobytes(),
            sample_rate=16000,
            num_channels=2,
            samples_per_channel=640,
        )
        result = mock_suppressor._process(frame)
        assert result.num_channels == 2
        assert result.samples_per_channel == 640


# ------------------------------------------------------------------ #
# Integration tests (require real model)                                #
# ------------------------------------------------------------------ #


def _model_available() -> bool:
    try:
        from livekit.plugins.hush._hush_model import HushModel

        HushModel()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _model_available(),
    reason="ONNX model not available",
)
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
        chunk = 5120  # one full chunk (32 frames)
        outputs = []

        for i in range(0, duration - chunk + 1, chunk):
            chunk_data = noisy[i : i + chunk]
            frame = create_audio_frame(chunk_data)
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

    def test_chunk_variable_sizes(self):
        """Model should handle different input lengths (padded to 32 frames)."""
        from livekit.plugins.hush._hush_model import HushModel, HushSession

        model = HushModel()
        session = HushSession(model)
        rng = np.random.default_rng(42)

        result = session.process_chunk(
            rng.uniform(-0.5, 0.5, 32 * 160).astype(np.float32)
        )
        assert len(result) == 32 * 160
        assert result.dtype == np.float32


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
        samples = rng.uniform(-0.5, 0.5, 5120).astype(np.float32)
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
        for _ in range(11):  # process 11 chunks to hit the log every 10
            samples = rng.uniform(-0.5, 0.5, 5120).astype(np.float32)
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

    def test_short_audio_returns_early(self, monkeypatch):
        """Audio shorter than hop_size (160 samples) returns immediately."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        # Override _CHUNK_SAMPLES back to real value for this test
        monkeypatch.setattr(ns_module, "_CHUNK_SAMPLES", 5120)

        # Push 80 samples (less than chunk) — passes through
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.5, 0.5, 80).astype(np.float32)
        frame = create_audio_frame(samples)
        result = ns._process(frame)
        out = np.frombuffer(result.data, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(out) == 80
        # Should be unmodified (still buffering)
        np.testing.assert_array_almost_equal(out, samples, decimal=4)

    def test_model_not_found_raises(self):
        """Nonexistent model path raises FileNotFoundError."""
        from livekit.plugins.hush._hush_model import HushModel

        with pytest.raises(FileNotFoundError):
            HushModel(model_path="/nonexistent/path.onnx")

    def test_process_chunk_2d_input(self):
        """Model handles 2D array input correctly."""
        from livekit.plugins.hush._hush_model import HushModel, HushSession

        model = HushModel()
        session = HushSession(model)
        rng = np.random.default_rng(99)
        audio_1d = rng.uniform(-0.5, 0.5, 32 * 160).astype(np.float32)
        audio_2d = audio_1d[np.newaxis, :]

        out_1d = session.process_chunk(audio_1d)
        out_2d = session.process_chunk(audio_2d)
        assert out_1d.shape == (32 * 160,)
        assert out_2d.shape == (1, 32 * 160)
        np.testing.assert_array_almost_equal(out_1d, out_2d[0], decimal=4)

    def test_process_chunk_too_short(self):
        """Audio shorter than hop_size returns unchanged."""
        from livekit.plugins.hush._hush_model import HushModel, HushSession

        model = HushModel()
        session = HushSession(model)
        short = np.array([0.1, -0.2, 0.05], dtype=np.float32)
        result = session.process_chunk(short)
        assert len(result) == 3
        np.testing.assert_array_equal(result, short)

    def test_trim_pad_output(self, monkeypatch):
        """Output trimming and padding paths when resampler length mismatches."""
        _patch_module(monkeypatch)
        import livekit.plugins.hush.noise_suppressor as ns_module

        # Test with 48kHz input to force resampling, which can cause
        # output/input length mismatch
        ns = ns_module.HushNoiseSuppressor(strength=1.0)
        rng = np.random.default_rng(42)

        # Generate enough samples for one full chunk after resampling
        samples = rng.uniform(-0.5, 0.5, 16000).astype(np.float32)
        int16_data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = rtc.AudioFrame(
            data=int16_data.tobytes(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=16000,
        )
        result = ns._process(frame)
        # Output length must match input length (trim/pad logic exercised)
        assert result.samples_per_channel == 16000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

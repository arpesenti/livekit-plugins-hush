"""Hush model inference using DeepFilterLib + ONNX Runtime.

The Hush model (pulp-vision/Hush) uses DeepFilterNet3 architecture with
ERB-domain feature extraction via the ``libdf`` C library and ONNX Runtime
for neural network inference. No PyTorch required at runtime.

Model sharing (Silero pattern):
    The ONNX session is loaded once per worker process and shared across
    all ``HushNoiseSuppressor`` instances. Each instance holds its own
    per-stream DF state and audio buffers. Memory footprint per worker:

        ~40 MB (onnxruntime + model) + N_sessions × ~1 MB (DF state + buffers)

Feature extraction via the ``libdf`` C library (pip: DeepFilterLib) is
auditable open-source from DeepFilterNet. ONNX Runtime is a widely-trusted
Microsoft inference engine available on PyPI.
"""

import logging
import os
import threading
from typing import Optional

import numpy as np
import onnxruntime as ort
from libdf import DF, erb, erb_norm, unit_norm

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_FFT_SIZE = 320
_HOP_SIZE = 160
_NB_ERB = 32
_NB_DF = 64
_NORM_TAU = 0.1
_CHUNK_FRAMES = 32  # fixed by ONNX model — T=32 baked into graph
_CHUNK_SAMPLES = _CHUNK_FRAMES * _HOP_SIZE  # 5120

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_DEFAULT_MODEL_FILE = "hush_dfnet_se.onnx"


def _compute_alpha(sr: int, hop: int, tau: float) -> float:
    return float(np.exp(-hop / (tau * sr)))


# ------------------------------------------------------------------ #
# Shared model (one per process)                                       #
# ------------------------------------------------------------------ #

_shared_model: Optional["HushModel"] = None
_shared_model_lock = threading.Lock()


def _get_shared_model(
    model_path: Optional[str] = None,
    atten_lim_db: float = 100.0,
) -> "HushModel":
    global _shared_model
    with _shared_model_lock:
        if _shared_model is None:
            _shared_model = HushModel(model_path, atten_lim_db)
        return _shared_model


class HushModel:
    """Shared ONNX model session — loaded once per worker process.

    This class holds only the ONNX Runtime inference session and
    model configuration. It does NOT hold per-stream DF state.
    Use :func:`_get_shared_model` to load or retrieve the singleton.

    Parameters
    ----------
    model_path : str, optional
        Path to the exported ONNX model file.
    atten_lim_db : float
        Maximum attenuation in dB (informational).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        atten_lim_db: float = 100.0,
    ) -> None:
        if model_path is None:
            model_path = os.path.join(_DEFAULT_MODEL_DIR, _DEFAULT_MODEL_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                "Run download_files() or re-export from PyTorch checkpoint."
            )

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._atten_lim_db = atten_lim_db

        # Warm-up: trigger ONNX Runtime JIT compilation once
        S = _CHUNK_FRAMES
        dummy_spec = np.zeros((1, 1, S, 161, 2), dtype=np.float32)
        dummy_erb = np.zeros((1, 1, S, 32), dtype=np.float32)
        dummy_spec_feat = np.zeros((1, 1, S, 64, 2), dtype=np.float32)
        self.session.run(
            None,
            {
                "spec": dummy_spec,
                "feat_erb": dummy_erb,
                "feat_spec": dummy_spec_feat,
            },
        )
        logger.debug("Hush ONNX model warm-up complete")


# ------------------------------------------------------------------ #
# Per-session state                                                    #
# ------------------------------------------------------------------ #


class HushSession:
    """Per-stream denoising session backed by a shared model.

    Holds the DeepFilterLib state (analysis/synthesis filters) and
    pre-computed alpha for feature normalization. Each audio stream
    (call) gets its own session.

    Parameters
    ----------
    model : HushModel
        The shared ONNX model session.
    """

    def __init__(self, model: HushModel) -> None:
        self._model = model
        self._session = model.session

        # Per-session DF state for feature extraction / synthesis
        self._df = DF(
            sr=_SAMPLE_RATE,
            fft_size=_FFT_SIZE,
            hop_size=_HOP_SIZE,
            nb_bands=_NB_ERB,
            min_nb_erb_freqs=2,
        )
        self._alpha = _compute_alpha(_SAMPLE_RATE, _HOP_SIZE, _NORM_TAU)

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Denoise a chunk of audio (32 frames = 5120 samples at 16 kHz).

        Parameters
        ----------
        audio : np.ndarray
            Float32 audio in [-1.0, 1.0], shape (samples,) or (1, samples).

        Returns
        -------
        np.ndarray
            Denoised audio, same shape as input.
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
            audio_squeezed = True
        else:
            audio_squeezed = False

        num_samples = audio.shape[1]
        S = num_samples // _HOP_SIZE

        if S != _CHUNK_FRAMES:
            raise ValueError(
                f"process_chunk requires exactly {_CHUNK_FRAMES} frames "
                f"({_CHUNK_SAMPLES} samples), got {S} frames ({num_samples} samples)"
            )

        # Pad for STFT: need fft_size extra samples
        padded_len = S * _HOP_SIZE + _FFT_SIZE
        if audio.shape[1] < padded_len:
            audio_padded = np.pad(audio, ((0, 0), (0, padded_len - audio.shape[1])))
        else:
            audio_padded = audio[:, :padded_len]

        # STFT analysis
        spec = self._df.analysis(audio_padded, reset=True)  # [1, S+2, 161]
        spec = spec[:, :S]

        # ERB and DF spectral features
        erb_feat = erb_norm(erb(spec, self._df.erb_widths()), self._alpha)  # [1, S, 32]
        spec_feat = unit_norm(spec[..., :_NB_DF], self._alpha)  # [1, S, 64] cplx

        # Convert to ONNX format [B, 1, T, F, 2]
        spec_in = np.stack([spec.real, spec.imag], axis=-1)[np.newaxis]
        feat_erb_in = erb_feat[:, np.newaxis]
        feat_spec_in = np.stack([spec_feat.real, spec_feat.imag], axis=-1)[np.newaxis]

        enhanced = self._session.run(
            None,
            {
                "spec": spec_in.astype(np.float32),
                "feat_erb": feat_erb_in.astype(np.float32),
                "feat_spec": feat_spec_in.astype(np.float32),
            },
        )[0]  # [1, 1, S, 161, 2]

        # Convert back to complex
        enhanced_c = enhanced[0, 0, :, :, 0] + 1j * enhanced[0, 0, :, :, 1]

        # ISTFT synthesis (expects 3D: [B, T, F])
        audio_out = self._df.synthesis(enhanced_c[np.newaxis, :, :], reset=True)

        # Compensate algorithmic delay
        delay = _FFT_SIZE - _HOP_SIZE
        output_len = num_samples
        if delay + output_len > audio_out.shape[1]:
            audio_out = np.pad(
                audio_out,
                ((0, 0), (0, delay + output_len - audio_out.shape[1])),
            )
        audio_out = audio_out[:, delay : delay + output_len]

        if audio_squeezed:
            return audio_out[0]
        return audio_out.reshape(audio.shape)

    def close(self) -> None:
        # Drop DF reference so tp_dealloc frees C-level analysis/synthesis buffers
        self._df = None
        self._session = None
        self._model = None

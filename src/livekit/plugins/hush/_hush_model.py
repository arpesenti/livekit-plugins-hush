"""Hush model inference using DeepFilterLib + ONNX Runtime.

The Hush model (pulp-vision/Hush) uses DeepFilterNet3 architecture split into
three ONNX sub-models (encoder, ERB decoder, DF decoder) that exactly match the
official C library's model bundle. ERB-domain feature extraction uses the
``libdf`` C library. No PyTorch required at runtime.

Model sharing (Silero pattern):
    The three ONNX sessions are loaded once per worker process and shared across
    all ``HushNoiseSuppressor`` instances. Each instance holds its own
    per-stream DF state and audio buffers. Memory footprint per worker:

        ~40 MB (onnxruntime + models) + N_sessions × ~1 MB (DF state + buffers)
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
_NORM_TAU = 1.0
_DF_ORDER = 5
_CHUNK_FRAMES = 32  # fixed by ONNX model — T=32 baked into graph
_CHUNK_SAMPLES = _CHUNK_FRAMES * _HOP_SIZE  # 5120

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _compute_alpha(sr: int, hop: int, tau: float) -> float:
    return float(np.exp(-hop / (tau * sr)))


# ------------------------------------------------------------------ #
# ERB filterbank builder (pure numpy, replicates PyTorch erb_fb)       #
# ------------------------------------------------------------------ #

def _build_erb_inv_fb() -> np.ndarray:
    """Build the inverse ERB filterbank matrix [32, 161] for mask application.

    Replicates ``DfNetSE.__init__`` filterbank construction from
    ``model/dfnet_se.py`` in the official Hush repository.
    """
    n_freqs = _FFT_SIZE // 2 + 1  # 161

    # Get ERB widths from libdf — same as compute_erb_widths()
    # (hop_size is unused for ERB width computation; using same value as
    #  the session for consistency)
    df_state = DF(
        sr=_SAMPLE_RATE,
        fft_size=_FFT_SIZE,
        hop_size=_HOP_SIZE,
        nb_bands=_NB_ERB,
        min_nb_erb_freqs=2,
    )
    widths = np.asarray(df_state.erb_widths(), dtype=np.int64)
    if widths.sum() != n_freqs:
        raise RuntimeError(
            f"libdf ERB widths sum to {widths.sum()}, expected {n_freqs}"
        )

    # Build rectangular ERB filterbank (forward) [161, 32]
    b_pts = np.cumsum(np.concatenate([[0], widths])).astype(int)[:-1]
    fb = np.zeros((n_freqs, _NB_ERB), dtype=np.float32)
    for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
        fb[b : b + w, i] = 1.0

    # Normalize columns (matching erb_fb with normalized=True)
    col_sum = fb.sum(axis=0)
    col_sum = np.maximum(col_sum, 1e-12)
    fb = fb / col_sum[np.newaxis, :]

    # Inverse filterbank: transpose [32, 161] (matching erb_fb with inverse=True)
    return fb.T.copy()


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
    """Shared ONNX model sessions — loaded once per worker process.

    Holds three ONNX Runtime inference sessions (encoder, ERB decoder,
    DF decoder) and the pre-built inverse ERB filterbank. Does NOT hold
    per-stream DF state.  Use :func:`_get_shared_model` to load or
    retrieve the singleton.

    Parameters
    ----------
    model_path : str, optional
        Path to the directory containing enc.onnx, erb_dec.onnx, df_dec.onnx.
    atten_lim_db : float
        Maximum attenuation in dB.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        atten_lim_db: float = 100.0,
    ) -> None:
        model_dir = model_path or _DEFAULT_MODEL_DIR

        enc_path = os.path.join(model_dir, "enc.onnx")
        erb_dec_path = os.path.join(model_dir, "erb_dec.onnx")
        df_dec_path = os.path.join(model_dir, "df_dec.onnx")

        for p in [enc_path, erb_dec_path, df_dec_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"ONNX model not found: {p}\n"
                    "Please ensure the three sub-model files are present."
                )

        self.enc_sess = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
        self.erb_dec_sess = ort.InferenceSession(erb_dec_path, providers=["CPUExecutionProvider"])
        self.df_dec_sess = ort.InferenceSession(df_dec_path, providers=["CPUExecutionProvider"])
        self.erb_inv_fb = _build_erb_inv_fb()
        self._atten_lim_db = atten_lim_db

        # Cache encoder output names (static after load)
        self._enc_output_names = [o.name for o in self.enc_sess.get_outputs()]

        # Warm-up: one dummy forward pass through all three sub-models
        S = _CHUNK_FRAMES
        dummy_erb = np.zeros((1, 1, S, _NB_ERB), dtype=np.float32)
        dummy_spec = np.zeros((1, 2, S, _NB_DF), dtype=np.float32)

        enc_out = self.enc_sess.run(None, {"feat_erb": dummy_erb, "feat_spec": dummy_spec})
        enc_dict = dict(zip(self._enc_output_names, enc_out))

        self.erb_dec_sess.run(
            None,
            {
                "emb": enc_dict["emb"],
                "e3": enc_dict["e3"],
                "e2": enc_dict["e2"],
                "e1": enc_dict["e1"],
                "e0": enc_dict["e0"],
            },
        )

        self.df_dec_sess.run(
            None,
            {"emb": enc_dict["emb"], "c0": enc_dict["c0"]},
        )
        logger.debug("Hush ONNX models (enc, erb_dec, df_dec) warm-up complete")


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
        The shared ONNX model sessions.
    """

    def __init__(self, model: HushModel) -> None:
        self._enc_sess = model.enc_sess
        self._erb_dec_sess = model.erb_dec_sess
        self._df_dec_sess = model.df_dec_sess
        self._enc_output_names = model._enc_output_names
        self._erb_inv_fb = model.erb_inv_fb  # [32, 161]
        self._atten_lim_db = model._atten_lim_db

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
        spec = spec[:, :S]  # [1, S, 161] complex

        # ERB and DF spectral features
        erb_feat = erb_norm(erb(spec, self._df.erb_widths()), self._alpha)  # [1, S, 32]
        spec_feat = unit_norm(spec[..., :_NB_DF], self._alpha)  # [1, S, 64] cplx

        # ---- Encoder ----
        # enc.onnx expects: feat_erb [1, 1, S, 32], feat_spec [1, 2, S, 64]
        enc_out = self._enc_sess.run(
            None,
            {
                "feat_erb": erb_feat[:, np.newaxis, :, :].astype(np.float32),
                "feat_spec": np.stack(
                    [spec_feat.real, spec_feat.imag], axis=1
                ).astype(np.float32),
            },
        )
        enc = dict(zip(self._enc_output_names, enc_out))
        # enc["emb"]: [1, S, 128]
        # enc["c0"]:  [1, 16, S, 64]
        # enc["e0"]:  [1, 16, S, 32]
        # enc["e1"]:  [1, 16, S, 16]
        # enc["e2"]:  [1, 16, S, 8]
        # enc["e3"]:  [1, 16, S, 8]

        # ---- ERB Decoder → mask ----
        mask = self._erb_dec_sess.run(
            None,
            {
                "emb": enc["emb"],
                "e3": enc["e3"],
                "e2": enc["e2"],
                "e1": enc["e1"],
                "e0": enc["e0"],
            },
        )[0]  # [1, 1, S, 32]

        # Apply ERB mask: project 32 bands → 161 freq bins via inverse filterbank
        # mask shape: [S, 32]; erb_inv_fb: [32, 161]; result: [S, 161]
        mask_2d = mask[0, 0]  # [S, 32]
        mask_projected = mask_2d @ self._erb_inv_fb  # [S, 32] @ [32, 161] = [S, 161]
        spec_masked = spec[0] * mask_projected  # [S, 161] * [S, 161] = [S, 161]

        # ---- DF Decoder → filter coefficients ----
        coefs_raw = self._df_dec_sess.run(
            None,
            {"emb": enc["emb"], "c0": enc["c0"]},
        )[0]  # [1, S, 64, 10]

        # Reshape from [1, S, 64, 10] to [1, _DF_ORDER, S, 64, 2]
        # The 10 values are 5 complex pairs: r0,i0, r1,i1, r2,i2, r3,i3, r4,i4
        coefs = coefs_raw.reshape(1, S, _NB_DF, _DF_ORDER, 2)
        coefs = coefs.transpose(0, 3, 1, 2, 4)  # [1, _DF_ORDER, S, 64, 2]

        # Apply DF filter to input spectrum (first 64 bins)
        # spec_input: [1, S, 161] complex → take nb_df bins → [1, S, 64, 2]
        spec_cplx = spec[0]  # [S, 161] complex
        spec_df = np.zeros((S, _NB_DF, 2), dtype=np.float32)
        spec_df[:, :, 0] = spec_cplx[:, :_NB_DF].real
        spec_df[:, :, 1] = spec_cplx[:, :_NB_DF].imag

        # Pad time dim: prepend _DF_ORDER - 1 frames (lookahead=0, causal)
        pad_frames = _DF_ORDER - 1
        spec_df_padded = np.pad(spec_df, ((pad_frames, 0), (0, 0), (0, 0)))
        # [S + pad_frames, 64, 2]

        # Unfold into windows of size _DF_ORDER along time axis
        windows = np.lib.stride_tricks.sliding_window_view(
            spec_df_padded, _DF_ORDER, axis=0
        )  # [S, 64, 2, _DF_ORDER]
        windows = windows.transpose(0, 3, 1, 2)  # [S, _DF_ORDER, 64, 2]

        # Complex multiply coefs @ windows
        # coefs: [1, _DF_ORDER, S, 64, 2] → squeeze batch → [_DF_ORDER, S, 64, 2]
        c = coefs[0]  # [_DF_ORDER, S, 64, 2]
        w = windows.transpose(1, 0, 2, 3)  # [_DF_ORDER, S, 64, 2]

        re = c[..., 0] * w[..., 0] - c[..., 1] * w[..., 1]  # [_DF_ORDER, S, 64]
        im = c[..., 1] * w[..., 0] + c[..., 0] * w[..., 1]  # [_DF_ORDER, S, 64]

        df_out_re = re.sum(axis=0)  # [S, 64]
        df_out_im = im.sum(axis=0)  # [S, 64]

        # Combine: DF-filtered for low bins, ERB-masked for high bins
        enhanced = spec_masked.copy()
        enhanced[:, :_NB_DF] = df_out_re + 1j * df_out_im

        # Apply attenuation limit
        if self._atten_lim_db < 100.0:
            input_mag = np.abs(spec_cplx)
            enhanced_mag = np.abs(enhanced)
            gain = enhanced_mag / (input_mag + 1e-10)
            min_gain = 10.0 ** (-self._atten_lim_db / 20.0)
            gain_limited = np.maximum(gain, min_gain)
            enhanced = enhanced * (gain_limited / (gain + 1e-10))

        # ISTFT synthesis (expects 3D: [B, T, F])
        audio_out = self._df.synthesis(
            enhanced.astype(np.complex64)[np.newaxis, :, :], reset=True
        )

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
        self._df = None
        self._enc_sess = None
        self._erb_dec_sess = None
        self._df_dec_sess = None

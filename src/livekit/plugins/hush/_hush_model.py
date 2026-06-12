"""Hush model inference using DeepFilterLib + ONNX Runtime.

Uses three ONNX sub-models (encoder, ERB decoder, DF decoder) from the
official C library bundle. Consecutive chunks are processed with
overlapping context so the GRU hidden state is seeded by previous frames,
eliminating rhythmic clicking at 32-frame boundaries.

Feature extraction uses the ``libdf`` C library. No PyTorch required.
"""

import logging, os, threading
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
_CHUNK_FRAMES = 32
_CHUNK_SAMPLES = _CHUNK_FRAMES * _HOP_SIZE

# Number of "warm-up" frames prepended to each chunk (except the first).
# These frames are discarded from the output but provide GRU context from
# the previous chunk. 12 frames provides a good trade-off between quality
# and overhead (~38% more computation).
_WARMUP_FRAMES = 12

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _compute_alpha(sr, hop, tau):
    return float(np.exp(-hop / (tau * sr)))


def _build_erb_inv_fb():
    n_freqs = _FFT_SIZE // 2 + 1
    df_state = DF(sr=_SAMPLE_RATE, fft_size=_FFT_SIZE, hop_size=_HOP_SIZE,
                  nb_bands=_NB_ERB, min_nb_erb_freqs=2)
    widths = np.asarray(df_state.erb_widths(), dtype=np.int64)
    if widths.sum() != n_freqs:
        raise RuntimeError(f"libdf ERB widths sum to {widths.sum()}, expected {n_freqs}")
    b_pts = np.cumsum(np.concatenate([[0], widths])).astype(int)[:-1]
    fb = np.zeros((n_freqs, _NB_ERB), dtype=np.float32)
    for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
        fb[b : b + w, i] = 1.0
    return fb.T.copy()


# ------------------------------------------------------------------ #
# Shared model (one per process)                                      #
# ------------------------------------------------------------------ #

_shared_model = None
_shared_model_lock = threading.Lock()


def _get_shared_model(model_path=None, atten_lim_db=100.0):
    global _shared_model
    with _shared_model_lock:
        if _shared_model is None:
            _shared_model = HushModel(model_path, atten_lim_db)
        return _shared_model


class HushModel:
    """Shared ONNX model sessions — loaded once per worker process."""

    def __init__(self, model_path=None, atten_lim_db=100.0):
        model_dir = model_path or _DEFAULT_MODEL_DIR

        enc_path = os.path.join(model_dir, "enc.onnx")
        erb_dec_path = os.path.join(model_dir, "erb_dec.onnx")
        df_dec_path = os.path.join(model_dir, "df_dec.onnx")

        for p in [enc_path, erb_dec_path, df_dec_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"ONNX model not found: {p}\n"
                    "Please ensure the sub-model files are present."
                )

        self.enc_sess = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
        self.erb_dec_sess = ort.InferenceSession(erb_dec_path, providers=["CPUExecutionProvider"])
        self.df_dec_sess = ort.InferenceSession(df_dec_path, providers=["CPUExecutionProvider"])
        self.erb_inv_fb = _build_erb_inv_fb()
        self._atten_lim_db = atten_lim_db

        self._enc_output_names = [o.name for o in self.enc_sess.get_outputs()]

        # Warm-up
        S = _CHUNK_FRAMES
        dummy_erb = np.zeros((1, 1, S, _NB_ERB), dtype=np.float32)
        dummy_spec = np.zeros((1, 2, S, _NB_DF), dtype=np.float32)
        enc_out = self.enc_sess.run(None, {"feat_erb": dummy_erb, "feat_spec": dummy_spec})
        enc_dict = dict(zip(self._enc_output_names, enc_out))
        self.erb_dec_sess.run(None, {
            "emb": enc_dict["emb"], "e3": enc_dict["e3"], "e2": enc_dict["e2"],
            "e1": enc_dict["e1"], "e0": enc_dict["e0"],
        })
        self.df_dec_sess.run(None, {"emb": enc_dict["emb"], "c0": enc_dict["c0"]})
        logger.debug("Hush ONNX models warm-up complete")


# ------------------------------------------------------------------ #
# Per-session state                                                   #
# ------------------------------------------------------------------ #

class HushSession:
    """Per-stream denoising session with overlapping-chunk GRU context."""

    def __init__(self, model):
        self._enc_sess = model.enc_sess
        self._erb_dec_sess = model.erb_dec_sess
        self._df_dec_sess = model.df_dec_sess
        self._enc_output_names = model._enc_output_names
        self._erb_inv_fb = model.erb_inv_fb
        self._atten_lim_db = model._atten_lim_db

        self._df = DF(sr=_SAMPLE_RATE, fft_size=_FFT_SIZE, hop_size=_HOP_SIZE,
                      nb_bands=_NB_ERB, min_nb_erb_freqs=2)
        self._alpha = _compute_alpha(_SAMPLE_RATE, _HOP_SIZE, _NORM_TAU)

        # Saved tail of the previous chunk for warm-up overlap
        self._prev_tail = None

    def _enhance_spectrum(self, spec_chunk, erb_chunk, sf_chunk,
                          prev_df_tail=None):
        """Run the model on one batch of spectrum frames.
        
        Returns (enhanced_spectrum, df_tail_for_next_chunk).
        """
        S = spec_chunk.shape[1]
        enc_out = self._enc_sess.run(None, {
            "feat_erb": erb_chunk[:, np.newaxis, :, :].astype(np.float32),
            "feat_spec": np.stack([sf_chunk.real, sf_chunk.imag], axis=1).astype(np.float32),
        })
        enc = dict(zip(self._enc_output_names, enc_out))

        mask = self._erb_dec_sess.run(None, {
            "emb": enc["emb"], "e3": enc["e3"], "e2": enc["e2"],
            "e1": enc["e1"], "e0": enc["e0"],
        })[0]
        spec_masked = spec_chunk[0] * (mask[0, 0] @ self._erb_inv_fb)

        coefs_raw = self._df_dec_sess.run(None, {"emb": enc["emb"], "c0": enc["c0"]})[0]
        coefs = coefs_raw.reshape(1, S, _NB_DF, _DF_ORDER, 2).transpose(0, 3, 1, 2, 4)

        spec_df = np.zeros((S, _NB_DF, 2), dtype=np.float32)
        spec_df[:, :, 0] = spec_chunk[0, :, :_NB_DF].real
        spec_df[:, :, 1] = spec_chunk[0, :, :_NB_DF].imag

        pf = _DF_ORDER - 1
        if prev_df_tail is not None and prev_df_tail.shape[0] == pf:
            spec_df_p = np.concatenate([prev_df_tail, spec_df], axis=0)
        else:
            spec_df_p = np.pad(spec_df, ((pf, 0), (0, 0), (0, 0)))
        next_df = spec_df[-pf:].copy()

        win = np.lib.stride_tricks.sliding_window_view(spec_df_p, _DF_ORDER, axis=0).transpose(0, 3, 1, 2)
        c = coefs[0]; w = win.transpose(1, 0, 2, 3)
        re = c[..., 0] * w[..., 0] - c[..., 1] * w[..., 1]
        im = c[..., 1] * w[..., 0] + c[..., 0] * w[..., 1]

        enhanced = spec_masked.copy()
        enhanced[:, :_NB_DF] = re.sum(axis=0) + 1j * im.sum(axis=0)
        return enhanced, next_df

    def process_chunk(self, audio):
        """Denoise a chunk of audio (32 frames = 5120 samples at 16 kHz).

        The first chunk is processed standalone. Subsequent chunks prepend
        warm-up frames from the previous chunk's tail so the GRU hidden
        state is seeded by recent audio context, eliminating boundary
        clicks.
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
                f"process_chunk requires {_CHUNK_FRAMES} frames, got {S} frames"
            )

        padded_len = S * _HOP_SIZE + _FFT_SIZE
        if audio.shape[1] < padded_len:
            audio_padded = np.pad(audio, ((0, 0), (0, padded_len - audio.shape[1])))
        else:
            audio_padded = audio[:, :padded_len]

        spec = self._df.analysis(audio_padded, reset=True)
        spec_model = spec[:, :S]
        spec_tail = spec[:, S : S + 1]

        erb_feat = erb_norm(erb(spec_model, self._df.erb_widths()), self._alpha)
        sf_feat = unit_norm(spec_model[..., :_NB_DF], self._alpha)

        # Prepend warm-up frames if available. We use the raw spectrum from
        # the previous chunk's tail so the GRU sees the actual audio context.
        # The warmup frames are re-normalized within the current chunk's
        # feature extraction call (necessary for consistent feature scaling).
        if self._prev_tail is not None and self._prev_tail.shape[1] > 0:
            W = min(self._prev_tail.shape[1], _WARMUP_FRAMES)
            warmup_spec = self._prev_tail[:, -W:]
            warmup_erb = erb_norm(erb(warmup_spec, self._df.erb_widths()), self._alpha)
            warmup_sf = unit_norm(warmup_spec[..., :_NB_DF], self._alpha)

            spec_batch = np.concatenate([warmup_spec, spec_model], axis=1)
            erb_batch = np.concatenate([warmup_erb, erb_feat], axis=1)
            sf_batch = np.concatenate([warmup_sf, sf_feat], axis=1)
        else:
            W = 0
            spec_batch = spec_model
            erb_batch = erb_feat
            sf_batch = sf_feat

        enhanced_batch, _ = self._enhance_spectrum(spec_batch, erb_batch, sf_batch)
        enhanced_model = enhanced_batch[W:]  # discard warm-up frames

        # Save tail for next chunk
        tail_len = _WARMUP_FRAMES
        if self._prev_tail is None:
            self._prev_tail = spec_model
        else:
            self._prev_tail = np.concatenate(
                [self._prev_tail, spec_model], axis=1
            )[:, -tail_len * 2:]  # keep up to 2x for safety

        # Build enhanced for synthesis.
        # Always reset=True per chunk — each chunk is STFT-independent.
        # Uses 33 frames (32 model + 1 zero tail) → 5280 samples.
        # Delay comp removes 160 → 5120 output.
        tail_frame = np.zeros((1, enhanced_model.shape[1]), dtype=np.complex64)
        enhanced = np.concatenate([enhanced_model, tail_frame], axis=0)
        audio_out = self._df.synthesis(
            enhanced.astype(np.complex64)[np.newaxis, :, :], reset=True
        )
        delay = _FFT_SIZE - _HOP_SIZE
        audio_out = audio_out[:, delay : delay + num_samples]

        if audio_squeezed:
            return audio_out[0]
        return audio_out.reshape(audio.shape)

    def reset_state(self):
        """Reset warm-up state for a new audio stream."""
        self._prev_tail = None

    def close(self):
        self._df = None
        self._enc_sess = None
        self._erb_dec_sess = None
        self._df_dec_sess = None
        self._prev_tail = None

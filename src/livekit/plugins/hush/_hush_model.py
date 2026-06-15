"""Hush model inference using DeepFilterLib + ONNX Runtime, per-frame streaming.

Matches the API shape of the upstream ``weya_nc`` C library: one 10 ms frame
in, one 10 ms frame out, with continuous GRU hidden state across calls.

The encoder, ERB decoder, and DF decoder each carry a SqueezedGRU whose
hidden state is exposed as an ONNX I/O. ``HushSession`` holds those three
states (and a 4-frame DF filter history) as plain numpy arrays, threading
them through every ``process_frame`` call.

Feature extraction uses the ``libdf`` C library, with ``reset=False`` so its
analysis and synthesis filter state is carried across frames. No PyTorch
required.

Re-exporting the ONNX sub-models with GRU state I/O: see
``scripts/export_onnx_stateful.py``.
"""

import logging
import os
import threading

import numpy as np
import onnxruntime as ort

from ._libdf import DF, erb, erb_norm, unit_norm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16_000
_FFT_SIZE = 320
_HOP_SIZE = 160
_FRAME_SAMPLES = _HOP_SIZE  # 160 samples = 10 ms at 16 kHz
_NB_ERB = 32
_NB_DF = 64
_NORM_TAU = 1.0
_DF_ORDER = 5

# GRU hidden state dimensions (must match the ONNX export)
_EMB_HIDDEN = 256
_DF_HIDDEN = 256
_ENC_NUM_LAYERS = 1
_ERB_DEC_NUM_LAYERS = 1
_DF_DEC_NUM_LAYERS = 3

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _compute_alpha(sr, hop, tau):
    return float(np.exp(-hop / (tau * sr)))


def _build_erb_inv_fb():
    n_freqs = _FFT_SIZE // 2 + 1
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
    b_pts = np.cumsum(np.concatenate([[0], widths])).astype(int)[:-1]
    freqs = np.arange(n_freqs)
    fb = ((freqs[:, None] >= b_pts) & (freqs[:, None] < b_pts + widths)).astype(
        np.float32
    )
    return fb.T.copy()


# ---------------------------------------------------------------------------
# Shared model (one per process)
# ---------------------------------------------------------------------------

_shared_model = None
_shared_model_lock = threading.Lock()


def _get_shared_model(model_path=None):
    global _shared_model
    with _shared_model_lock:
        if _shared_model is None:
            _shared_model = HushModel(model_path)
        return _shared_model


def _make_low_latency_session_options():
    """ORTT session options tuned for low-latency single-stream inference.

    Mirrors the silero VAD plugin's config: single thread per op, no
    inter-op parallelism, no spinning waits, sequential execution mode.
    Yields a ~2x per-frame speedup over ORTT defaults on the Hush
    sub-models because it avoids the per-op thread-pool overhead that
    onnxruntime enables by default for parallel ops.

    Graph optimization is enabled to fuse constant subgraphs and
    eliminate redundant transposes; this is a free 10-15% speedup
    over the default (which is ORT_ENABLE_BASIC).
    """
    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


class HushModel:
    """Shared ONNX model sessions — loaded once per worker process."""

    def __init__(self, model_path=None):
        model_dir = model_path or _DEFAULT_MODEL_DIR

        enc_path = os.path.join(model_dir, "enc.onnx")
        erb_dec_path = os.path.join(model_dir, "erb_dec.onnx")
        df_dec_path = os.path.join(model_dir, "df_dec.onnx")

        for p in [enc_path, erb_dec_path, df_dec_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"ONNX model not found: {p}\n"
                    "Please ensure the sub-model files are present. "
                    "Re-export with scripts/export_onnx_stateful.py if needed."
                )

        sess_opts = _make_low_latency_session_options()
        self.enc_sess = ort.InferenceSession(
            enc_path, providers=["CPUExecutionProvider"], sess_options=sess_opts
        )
        self.erb_dec_sess = ort.InferenceSession(
            erb_dec_path, providers=["CPUExecutionProvider"], sess_options=sess_opts
        )
        self.df_dec_sess = ort.InferenceSession(
            df_dec_path, providers=["CPUExecutionProvider"], sess_options=sess_opts
        )
        self.erb_inv_fb = _build_erb_inv_fb()

        # Warm-up: trigger ONNX Runtime JIT compilation once. Use a
        # 1-frame input with zero hidden state.
        s = 1
        self.enc_sess.run(
            None,
            {
                "feat_erb": np.zeros((1, 1, s, _NB_ERB), dtype=np.float32),
                "feat_spec": np.zeros((1, 2, s, _NB_DF), dtype=np.float32),
                "h_enc_in": np.zeros(
                    (_ENC_NUM_LAYERS, 1, _EMB_HIDDEN), dtype=np.float32
                ),
            },
        )
        self.erb_dec_sess.run(
            None,
            {
                "emb": np.zeros((1, s, 128), dtype=np.float32),
                "e3": np.zeros((1, 16, s, 8), dtype=np.float32),
                "e2": np.zeros((1, 16, s, 8), dtype=np.float32),
                "e1": np.zeros((1, 16, s, 16), dtype=np.float32),
                "e0": np.zeros((1, 16, s, 32), dtype=np.float32),
                "h_erb_dec_in": np.zeros(
                    (_ERB_DEC_NUM_LAYERS, 1, _EMB_HIDDEN), dtype=np.float32
                ),
            },
        )
        self.df_dec_sess.run(
            None,
            {
                "emb": np.zeros((1, s, 128), dtype=np.float32),
                "c0": np.zeros((1, 16, s, _NB_DF), dtype=np.float32),
                "h_df_dec_in": np.zeros(
                    (_DF_DEC_NUM_LAYERS, 1, _DF_HIDDEN), dtype=np.float32
                ),
            },
        )
        logger.debug("Hush ONNX models warm-up complete")


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


class HushSession:
    """Per-stream denoising session, matching the C library's API shape.

    One frame of 160 samples (10 ms) in, one frame out, with continuous
    GRU hidden state and DF filter history across calls. ``reset_state()``
    clears all state.
    """

    def __init__(self, model, atten_lim_db: float = 100.0):
        self._enc_sess = model.enc_sess
        self._erb_dec_sess = model.erb_dec_sess
        self._df_dec_sess = model.df_dec_sess
        self._erb_inv_fb = model.erb_inv_fb

        # libdf state — runs in streaming mode (reset=False) so the
        # analysis/synthesis filter state is carried across frames.
        self._df = DF(
            sr=_SAMPLE_RATE,
            fft_size=_FFT_SIZE,
            hop_size=_HOP_SIZE,
            nb_bands=_NB_ERB,
            min_nb_erb_freqs=2,
        )
        self._alpha = _compute_alpha(_SAMPLE_RATE, _HOP_SIZE, _NORM_TAU)

        # Precompute the linear-blend attenuation coefficient. The
        # upstream reference (`scripts/infer_single.py` in pulp-vision/Hush)
        # does:  spec_out = spec_in * lim + spec_enh * (1.0 - lim)
        # where lim = 10**(-atten_lim_db / 20). lim=1.0 → passthrough,
        # lim=0.0 → full model output. Default 100.0 dB → lim ≈ 1e-5,
        # effectively a passthrough of the model output.
        if atten_lim_db < 100.0:
            self._lim = 10.0 ** (-atten_lim_db / 20.0)
        else:
            self._lim = 0.0

        # State: all zeroed on init / reset. Shapes:
        #   _h_enc:      [ENC_NUM_LAYERS, 1, EMB_HIDDEN]
        #   _h_erb_dec:  [ERB_DEC_NUM_LAYERS, 1, EMB_HIDDEN]
        #   _h_df_dec:   [DF_DEC_NUM_LAYERS, 1, DF_HIDDEN]
        #   _prev_df:    [_DF_ORDER-1, _NB_DF, 2] float32 (DF filter history)
        self._reset_state()

    def _reset_state(self):
        self._h_enc = np.zeros((_ENC_NUM_LAYERS, 1, _EMB_HIDDEN), dtype=np.float32)
        self._h_erb_dec = np.zeros(
            (_ERB_DEC_NUM_LAYERS, 1, _EMB_HIDDEN), dtype=np.float32
        )
        self._h_df_dec = np.zeros((_DF_DEC_NUM_LAYERS, 1, _DF_HIDDEN), dtype=np.float32)
        self._prev_df = np.zeros((_DF_ORDER - 1, _NB_DF, 2), dtype=np.float32)
        # Pre-allocated per-frame scratch buffers. Avoids the per-frame
        # np.zeros + np.concatenate + np.copyto allocations on the hot path.
        self._spec_df_new = np.empty((1, _NB_DF, 2), dtype=np.float32)
        self._spec_df_p = np.empty((_DF_ORDER, _NB_DF, 2), dtype=np.float32)
        self._feat_spec = np.empty((1, 2, 1, _NB_DF), dtype=np.float32)
        # Reset libdf's analysis/synthesis filter state so the next
        # audio stream starts from a clean STFT. The first 10 ms
        # after reset will be the STFT warmup (output near zero).
        if self._df is not None:
            self._df.reset()

    def reset_state(self) -> None:
        """Reset all per-stream state for a new audio source.

        Clears the encoder/decoder GRU hidden states, the DF filter
        history, and the libdf STFT filter state. The first 10 ms of
        audio after a reset will be the STFT warmup (output near zero);
        this matches the C library's ``weya_nc_reset`` behavior.
        """
        self._reset_state()

    def close(self) -> None:
        self._df = None
        self._enc_sess = None
        self._erb_dec_sess = None
        self._df_dec_sess = None
        self._h_enc = None
        self._h_erb_dec = None
        self._h_df_dec = None
        self._prev_df = None

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def process_frame(self, audio: np.ndarray) -> np.ndarray:
        """Denoise a single 160-sample (10 ms) frame at 16 kHz.

        The libdf analysis state, the encoder/decoder GRU hidden states, and
        the DF filter polynomial history are all carried across calls.
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
            squeezed = True
        else:
            squeezed = False

        if audio.shape[1] != _FRAME_SAMPLES:
            raise ValueError(
                f"process_frame requires {_FRAME_SAMPLES} samples, got {audio.shape[1]}"
            )

        # ---- STFT (streaming) -----------------------------------------
        # libdf.analysis(audio, reset=False) keeps the analysis filter
        # state across calls. For 160 samples with FFT=320, it produces
        # 1 frame.
        if audio.dtype is np.float32:
            spec_new = self._df.analysis(audio, reset=False)
        else:
            spec_new = self._df.analysis(audio.astype(np.float32), reset=False)
        # spec_new: (1, 1, 161) complex64

        # ---- Feature extraction (per frame) ----------------------------
        # The DF class owns the EMA state for both normalizations
        # (matches libdf's in-place semantics). On a fresh session
        # the state is None and gets initialized to the libdf defaults.
        erb_feat, self._df._erb_norm_state = erb_norm(
            erb(spec_new, self._df.erb_widths()),
            self._alpha,
            self._df._erb_norm_state,
        )  # (1, 1, 32)
        sf_feat, self._df._unit_norm_state = unit_norm(
            spec_new[..., :_NB_DF].copy(), self._alpha, self._df._unit_norm_state
        )  # (1, 1, 64) complex

        # ---- Encoder ----------------------------------------------------
        # Single-frame input. The GRU hidden state carries context.
        # Use pre-allocated feat_spec buffer (real, imag) instead of
        # allocating a fresh np.stack every frame.
        self._feat_spec[0, 0, 0, :] = sf_feat.real[0, 0, :]
        self._feat_spec[0, 1, 0, :] = sf_feat.imag[0, 0, :]
        enc_out = self._enc_sess.run(
            None,
            {
                "feat_erb": erb_feat[:, np.newaxis, :, :],
                "feat_spec": self._feat_spec,
                "h_enc_in": self._h_enc,
            },
        )
        e0, e1, e2, e3, emb, c0, _lsnr, self._h_enc = enc_out
        # All outputs shape (1, 1, ...) for the single time step.

        # ---- ERB decoder ------------------------------------------------
        m, self._h_erb_dec = self._erb_dec_sess.run(
            None,
            {
                "emb": emb,
                "e3": e3,
                "e2": e2,
                "e1": e1,
                "e0": e0,
                "h_erb_dec_in": self._h_erb_dec,
            },
        )
        # m: (1, 1, 1, 32) — gain mask per ERB band

        # ---- DF decoder -------------------------------------------------
        coefs, self._h_df_dec = self._df_dec_sess.run(
            None,
            {
                "emb": emb,
                "c0": c0,
                "h_df_dec_in": self._h_df_dec,
            },
        )
        # coefs: (1, 1, 64, 10) — DF filter per freq bin

        # ---- Post-process spectrum --------------------------------------
        spec_in = spec_new[0, 0]  # (161,) complex64
        mask = m[0, 0, 0]  # (32,) float32
        coef = coefs[0, 0]  # (64, 10) float32

        # Project ERB mask to full spectrum.
        spec_masked = spec_in * (mask @ self._erb_inv_fb)  # (161,) complex

        # Build DF filter window. The 5-tap polynomial prediction needs
        # 4 frames of "previous filter history" + the new frame.
        # _prev_df holds the 4 frames of (real, imag) saved from the
        # previous call's spec_df_p (or zeros on the first call).
        # Use pre-allocated scratch buffers to avoid per-frame allocations.
        self._spec_df_new[0, :, 0] = spec_in[:_NB_DF].real
        self._spec_df_new[0, :, 1] = spec_in[:_NB_DF].imag
        # Roll the prev frames down and append the new frame
        self._spec_df_p[:-1] = self._prev_df
        self._spec_df_p[-1:] = self._spec_df_new
        # shape: (5, 64, 2)

        # Save the last 4 frames for the next call.
        np.copyto(self._prev_df, self._spec_df_p[1:])

        # Apply the 5-tap complex FIR.
        # coef from ONNX: (64, 10) = (F, O*2). PyTorch reference does
        # coefs.permute(0, 2, 1, 3, 4) to put the order axis first →
        # (B, T, O, F, 2). We need c with shape (O, F, 2).
        c = coef.reshape(_NB_DF, _DF_ORDER, 2).transpose(1, 0, 2)
        # spec_df_p: (5, 64, 2). y[f] = sum_t c[t, f, 0]*x[t, f, 0] - c[t, f, 1]*x[t, f, 1]
        re = (c[..., 0] * self._spec_df_p[..., 0] - c[..., 1] * self._spec_df_p[..., 1]).sum(axis=0)
        im = (c[..., 1] * self._spec_df_p[..., 0] + c[..., 0] * self._spec_df_p[..., 1]).sum(axis=0)

        # Write the DF output directly into spec_masked (no extra copy).
        spec_masked[:_NB_DF] = re + 1j * im
        enhanced = spec_masked

        # ---- Attenuation limit (linear blend, per reference) -----------
        # Skip the multiply when atten_lim_db is 100 (no blending needed).
        if self._lim > 0.0:
            enhanced = spec_in * self._lim + enhanced * (1.0 - self._lim)

        # ---- STFT synthesis (streaming) --------------------------------
        # libdf.synthesis with 1 frame gives 160 samples with reset=False.
        audio_out = self._df.synthesis(enhanced[np.newaxis, np.newaxis, :], reset=False)
        # audio_out: (1, 160) float32

        if squeezed:
            return audio_out[0]
        return audio_out.reshape(audio.shape)

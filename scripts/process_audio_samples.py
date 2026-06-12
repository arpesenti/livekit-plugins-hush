"""Process original WAV files through Hush noise suppression.

Generates two denoised variants:
  hush-{name}.wav        — batch mode (full-file normalization + synthesis)
  hush-stream-{name}.wav  — streaming mode (HushSession.process_chunk)
"""

import sys, os, wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from livekit.plugins.hush._hush_model import (
    HushModel, HushSession,
    _SAMPLE_RATE, _FFT_SIZE, _HOP_SIZE, _NB_DF, _NB_ERB, _DF_ORDER,
    _CHUNK_FRAMES, _CHUNK_SAMPLES,
    _build_erb_inv_fb, _compute_alpha, _NORM_TAU,
)
from libdf import DF, erb, erb_norm, unit_norm


def read_wav(path):
    with wave.open(path, "rb") as wf:
        nc, sw, sr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(nf)
    if sw == 1:
        s = np.frombuffer(raw, np.uint8).astype(np.float32) / 255.0 * 2.0 - 1.0
    elif sw == 2:
        s = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    else:
        s = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    if nc > 1:
        s = s.reshape(-1, nc).mean(axis=1)
    return s, sr


def write_wav(path, samples, sr):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    i16 = np.clip(samples, -1.0, 1.0) * 32767.0
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(i16.astype(np.int16).tobytes())


def _model_chunk(spec_chunk, erb_chunk, sf_chunk, model, erb_inv_fb, prev_df):
    """Run one chunk through the model."""
    enc_out = model.enc_sess.run(None, {
        "feat_erb": erb_chunk[:, np.newaxis, :, :].astype(np.float32),
        "feat_spec": np.stack([sf_chunk.real, sf_chunk.imag], axis=1).astype(np.float32),
    })
    enc = dict(zip(model._enc_output_names, enc_out))
    mask = model.erb_dec_sess.run(None, {
        "emb": enc["emb"], "e3": enc["e3"], "e2": enc["e2"],
        "e1": enc["e1"], "e0": enc["e0"],
    })[0]
    spec_masked = spec_chunk[0] * (mask[0, 0] @ erb_inv_fb)
    coefs_raw = model.df_dec_sess.run(None, {"emb": enc["emb"], "c0": enc["c0"]})[0]
    coefs = coefs_raw.reshape(1, _CHUNK_FRAMES, _NB_DF, _DF_ORDER, 2).transpose(0, 3, 1, 2, 4)
    spec_df = np.zeros((_CHUNK_FRAMES, _NB_DF, 2), dtype=np.float32)
    spec_cplx = spec_chunk[0]
    spec_df[:, :, 0] = spec_cplx[:, :_NB_DF].real
    spec_df[:, :, 1] = spec_cplx[:, :_NB_DF].imag
    pf = _DF_ORDER - 1
    if prev_df is not None and prev_df.shape[0] == pf:
        spec_df_p = np.concatenate([prev_df, spec_df], axis=0)
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


def process_batch(audio, model, alpha, erb_inv_fb):
    """Full-file normalization + single model pass. Matches infer_single.py."""
    df = DF(sr=_SAMPLE_RATE, fft_size=_FFT_SIZE, hop_size=_HOP_SIZE,
            nb_bands=_NB_ERB, min_nb_erb_freqs=2)
    padded = np.pad(audio, (0, _FFT_SIZE)).astype(np.float32)
    spec_full = df.analysis(padded[np.newaxis, :], reset=True)
    spec_all = spec_full[:, :-1]  # all frames except the last tail frame
    n_frames = spec_all.shape[1]
    erb_all = erb_norm(erb(spec_all, df.erb_widths()), alpha)
    sf_all = unit_norm(spec_all[..., :_NB_DF], alpha)

    # Single forward pass through the model — GRU state is continuous
    # across all frames (exactly like infer_single.py).
    enc_out = model.enc_sess.run(None, {
        "feat_erb": erb_all[:, np.newaxis, :, :].astype(np.float32),
        "feat_spec": np.stack([sf_all.real, sf_all.imag], axis=1).astype(np.float32),
    })
    enc = dict(zip(model._enc_output_names, enc_out))

    mask = model.erb_dec_sess.run(None, {
        "emb": enc["emb"], "e3": enc["e3"], "e2": enc["e2"],
        "e1": enc["e1"], "e0": enc["e0"],
    })[0]

    coefs_raw = model.df_dec_sess.run(None, {
        "emb": enc["emb"], "c0": enc["c0"],
    })[0]  # [1, n_frames, 64, 10]
    coefs = coefs_raw.reshape(1, n_frames, _NB_DF, _DF_ORDER, 2).transpose(0, 3, 1, 2, 4)

    # Apply ERB mask (all frames at once)
    mask_proj = mask[0, 0] @ erb_inv_fb  # [n_frames, 32] @ [32, 161]
    spec_masked = spec_all[0] * mask_proj  # [n_frames, 161]

    # Apply DF filter (all frames at once)
    spec_df = np.zeros((n_frames, _NB_DF, 2), dtype=np.float32)
    spec_df[:, :, 0] = spec_all[0, :, :_NB_DF].real
    spec_df[:, :, 1] = spec_all[0, :, :_NB_DF].imag
    spec_df_p = np.pad(spec_df, ((_DF_ORDER - 1, 0), (0, 0), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(spec_df_p, _DF_ORDER, axis=0).transpose(0, 3, 1, 2)
    c = coefs[0]; w = win.transpose(1, 0, 2, 3)
    re = c[..., 0] * w[..., 0] - c[..., 1] * w[..., 1]
    im = c[..., 1] * w[..., 0] + c[..., 0] * w[..., 1]
    enhanced = spec_masked.copy()
    enhanced[:, :_NB_DF] = re.sum(axis=0) + 1j * im.sum(axis=0)

    # Full-file synthesis
    tail = np.zeros((1, enhanced.shape[1]), dtype=np.complex64)
    enhanced_full = np.concatenate([enhanced, tail], axis=0)
    audio_out = df.synthesis(enhanced_full[np.newaxis, :, :], reset=True)
    delay = _FFT_SIZE - _HOP_SIZE
    return audio_out[0, delay : delay + len(audio)]


def process_stream(audio, model):
    """HushSession.process_chunk — real-time quality with overlap."""
    session = HushSession(model)
    output = np.empty(len(audio), dtype=np.float32)
    pos = 0
    while pos < len(audio):
        end = min(pos + _CHUNK_SAMPLES, len(audio))
        chunk = audio[pos:end]
        if len(chunk) < _CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, _CHUNK_SAMPLES - len(chunk)))
        denoised = session.process_chunk(chunk)
        n_out = min(len(denoised), end - pos)
        output[pos : pos + n_out] = denoised[:n_out]
        pos += _CHUNK_SAMPLES
    session.close()
    return output


def main():
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "audio")
    originals_dir = os.path.join(docs_dir, "originals")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "src", "livekit", "plugins", "hush", "models")

    print("Loading Hush model...")
    model = HushModel(model_dir)
    alpha = _compute_alpha(_SAMPLE_RATE, _HOP_SIZE, _NORM_TAU)
    erb_inv_fb = _build_erb_inv_fb()

    for fname in sorted(os.listdir(originals_dir)):
        if not fname.endswith(".wav"):
            continue
        print(f"Processing {fname} ...")
        audio_data, sr = read_wav(os.path.join(originals_dir, fname))
        if sr != _SAMPLE_RATE:
            raise ValueError(f"Expected {_SAMPLE_RATE} Hz, got {sr} Hz")

        write_wav(os.path.join(docs_dir, f"hush-{fname.replace('.wav','')}.wav"),
                  process_batch(audio_data, model, alpha, erb_inv_fb), _SAMPLE_RATE)
        write_wav(os.path.join(docs_dir, f"hush-stream-{fname.replace('.wav','')}.wav"),
                  process_stream(audio_data, model), _SAMPLE_RATE)
        print(f"  Done: {fname}")

    print("All files processed.")


if __name__ == "__main__":
    main()

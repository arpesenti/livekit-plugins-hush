"""Comparison test: our ONNX pipeline vs. original PyTorch Hush model.

Verifies the exported ONNX model produces identical output to the original
PyTorch implementation. Requires PyTorch — install with:

    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

Run:

    python scripts/verify_onnx_export.py
"""

import os
import sys

import numpy as np


def _dependencies_available() -> bool:
    try:
        import torch  # noqa: F401
        from libdf import DF, erb, erb_norm, unit_norm  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def test_onnx_vs_pytorch() -> None:
    """Feed identical audio through both pipelines, compare output."""
    import torch
    from libdf import DF, erb, erb_norm, unit_norm
    import onnxruntime as ort

    # Add Hush repo to path (assumes it's checked out at /tmp/Hush)
    hush_repo = os.environ.get("HUSH_REPO", "/tmp/Hush")
    if not os.path.isdir(hush_repo):
        print(f"Hush repo not found at {hush_repo}. Clone with:")
        print(f"  git clone https://github.com/pulp-vision/Hush.git {hush_repo}")
        sys.exit(1)

    sys.path.insert(0, hush_repo)
    from model.dfnet_se import DfNetSE, get_config, as_real, as_complex, get_norm_alpha

    # --- Load both models ---

    # PyTorch
    config = get_config()
    pt_model = DfNetSE(config)
    ckpt_path = os.environ.get("HUSH_CKPT", "model_best.ckpt")
    if not os.path.exists(ckpt_path):
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download("weya-ai/hush", "model_best.ckpt")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    pt_model.model.load_state_dict(checkpoint)
    pt_model.eval()

    # ONNX
    onnx_path = os.environ.get(
        "HUSH_ONNX",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "livekit",
            "plugins",
            "hush",
            "models",
            "hush_dfnet_se.onnx",
        ),
    )
    onnx_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # --- Generate test audio ---
    S = 32
    rng = np.random.default_rng(42)
    df_state = DF(sr=16000, fft_size=320, hop_size=160, nb_bands=32, min_nb_erb_freqs=2)
    alpha = get_norm_alpha(16000, 160, config.norm_tau)

    audio = rng.uniform(-0.5, 0.5, S * 160 + 320).astype(np.float32)
    audio = audio[np.newaxis, :]

    # --- Shared feature extraction ---
    spec_np = df_state.analysis(audio, reset=True)
    spec_np = spec_np[:, :S]
    erb_feat_np = erb_norm(erb(spec_np, df_state.erb_widths()), alpha)
    spec_feat_np = unit_norm(spec_np[..., :64], alpha)

    # --- PyTorch inference ---
    spec_t = as_real(torch.as_tensor(spec_np)).unsqueeze(1)
    erb_feat_t = torch.as_tensor(erb_feat_np).unsqueeze(1)
    spec_feat_t = as_real(torch.as_tensor(spec_feat_np)).unsqueeze(1)

    with torch.no_grad():
        enhanced_pt = pt_model.model(spec_t.clone(), erb_feat_t, spec_feat_t)[0]
        enhanced_pt_np = enhanced_pt.numpy()

    # --- ONNX inference ---
    spec_in = np.stack([spec_np.real, spec_np.imag], axis=-1)[np.newaxis]
    erb_in = erb_feat_np[:, np.newaxis]
    spec_feat_in = np.stack([spec_feat_np.real, spec_feat_np.imag], axis=-1)[np.newaxis]

    enhanced_onnx = onnx_sess.run(
        None,
        {
            "spec": spec_in.astype(np.float32),
            "feat_erb": erb_in.astype(np.float32),
            "feat_spec": spec_feat_in.astype(np.float32),
        },
    )[0]

    # --- Compare ---
    diff = np.abs(enhanced_onnx - enhanced_pt_np)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    print(f"Shapes: ONNX {enhanced_onnx.shape} vs PyTorch {enhanced_pt_np.shape}")
    print(f"Max absolute difference:  {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    # Both should produce the enhanced spectrum
    assert enhanced_onnx.shape == enhanced_pt_np.shape, "Shape mismatch"
    assert max_diff < 5e-3, f"Max diff {max_diff:.6f} exceeds 5e-3 threshold"

    # --- Verify audio output matches ---
    enhanced_onnx_c = enhanced_onnx[0, 0, :, :, 0] + 1j * enhanced_onnx[0, 0, :, :, 1]
    # PyTorch output: [1, 1, S, 161, 2] → squeeze to [S, 161] complex
    enhanced_pt_c = as_complex(
        torch.as_tensor(enhanced_pt_np).squeeze(0).squeeze(0)
    ).numpy()  # [S, 161] complex

    audio_onnx = df_state.synthesis(enhanced_onnx_c[np.newaxis], reset=True)
    audio_pt = df_state.synthesis(enhanced_pt_c[np.newaxis], reset=True)

    audio_diff = np.abs(audio_onnx - audio_pt)
    max_audio_diff = float(audio_diff.max())
    print(f"Max audio diff after synthesis: {max_audio_diff:.8f}")

    delay = 320 - 160
    audio_onnx = audio_onnx[:, delay : delay + S * 160]
    audio_pt = audio_pt[:, delay : delay + S * 160]

    rms_onnx = float(np.sqrt(np.mean(audio_onnx**2)))
    rms_pt = float(np.sqrt(np.mean(audio_pt**2)))
    print(f"Output RMS: ONNX {rms_onnx:.6f}  PyTorch {rms_pt:.6f}")

    rms_ratio = rms_onnx / rms_pt if rms_pt > 0 else 0.0
    assert 0.97 < rms_ratio < 1.03, f"RMS ratio {rms_ratio:.6f} outside tolerance"

    print("PASS: ONNX output matches PyTorch output")
    return True


if __name__ == "__main__":
    if not _dependencies_available():
        print("SKIP: PyTorch not installed. Install with:")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(0)
    test_onnx_vs_pytorch()

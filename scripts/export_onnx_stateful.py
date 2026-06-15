"""Re-export Hush ONNX sub-models with GRU hidden state as I/O.

Exports three sub-models (enc, erb_dec, df_dec) where each internal
SqueezedGRU takes its previous hidden state as an input and emits the
new one as an output. The hidden state is shaped [num_layers, B, H].

Output:
    src/livekit/plugins/hush/models/enc.onnx
    src/livekit/plugins/hush/models/erb_dec.onnx
    src/livekit/plugins/hush/models/df_dec.onnx

Dependencies (only needed for re-export, not for installing or using
the plugin; not in pyproject.toml):
    pip install torch onnx onnxscript onnxruntime huggingface_hub

Usage:
    python scripts/export_onnx_stateful.py

The script clones the pulp-vision/Hush repo to /tmp/Hush (if not
already present), downloads the pretrained PyTorch weights from
Hugging Face, and re-exports the three sub-models with the GRU
hidden state exposed as I/O.
"""

import os
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn

HUSH_REPO_URL = "https://github.com/pulp-vision/Hush.git"
HUSH_REPO_DIR = "/tmp/Hush"
HUGGINGFACE_REPO = "weya-ai/hush"
HUGGINGFACE_FILE = "model_best.ckpt"


def _ensure_hush_repo():
    """Clone the pulp-vision/Hush repo if not already present."""
    if os.path.isdir(os.path.join(HUSH_REPO_DIR, "model")):
        sys.path.insert(0, HUSH_REPO_DIR)
        return
    print(f"Cloning {HUSH_REPO_URL} to {HUSH_REPO_DIR}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", HUSH_REPO_URL, HUSH_REPO_DIR],
        check=True,
    )
    sys.path.insert(0, HUSH_REPO_DIR)


def _ensure_hush_weights():
    """Download the pretrained PyTorch weights from Hugging Face."""
    from huggingface_hub import hf_hub_download

    cache = os.path.join(tempfile.gettempdir(), "hush_export")
    return hf_hub_download(HUGGINGFACE_REPO, HUGGINGFACE_FILE, cache_dir=cache)


_ensure_hush_repo()
from model.dfnet_se import DfNetSE, get_config  # noqa: E402

# GRU hidden state dimensions (from ModelConfig defaults)
EMB_HIDDEN = 256
DF_HIDDEN = 256
EMB_OUT = 128  # conv_ch * nb_erb // 4 = 16 * 32 // 4

ENC_NUM_LAYERS = 1
ERB_DEC_NUM_LAYERS = 1  # emb_num_layers - 1
DF_DEC_NUM_LAYERS = 3


class StatefulEncoder(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feat_erb, feat_spec, h_in):
        e0 = self.enc.erb_conv0(feat_erb)
        e1 = self.enc.erb_conv1(e0)
        e2 = self.enc.erb_conv2(e1)
        e3 = self.enc.erb_conv3(e2)
        c0 = self.enc.df_conv0(feat_spec)
        c1 = self.enc.df_conv1(c0)

        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = self.enc.df_fc_emb(cemb)
        emb_pre = e3.permute(0, 2, 3, 1).flatten(2)
        emb_pre = self.enc.combine(emb_pre, cemb)

        emb, h_out = self.enc.emb_gru(emb_pre, h_in)
        lsnr = self.enc.lsnr_fc(emb) * self.enc.lsnr_scale + self.enc.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr, h_out


class StatefulErbDecoder(nn.Module):
    def __init__(self, erb_dec):
        super().__init__()
        self.erb_dec = erb_dec

    def forward(self, emb, e3, e2, e1, e0, h_in):
        b, _, t, f8 = e3.shape
        emb, h_out = self.erb_dec.emb_gru(emb, h_in)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)
        e3 = self.erb_dec.convt3(self.erb_dec.conv3p(e3) + emb)
        e2 = self.erb_dec.convt2(self.erb_dec.conv2p(e2) + e3)
        e1 = self.erb_dec.convt1(self.erb_dec.conv1p(e1) + e2)
        m = self.erb_dec.conv0_out(self.erb_dec.conv0p(e0) + e1)
        return m, h_out


class StatefulDfDecoder(nn.Module):
    def __init__(self, df_dec):
        super().__init__()
        self.df_dec = df_dec

    def forward(self, emb, c0, h_in):
        b, t, _ = emb.shape
        c, h_out = self.df_dec.df_gru(emb, h_in)
        if self.df_dec.df_skip is not None:
            c = c + self.df_dec.df_skip(emb)
        c0 = self.df_dec.df_convp(c0).permute(0, 2, 3, 1)
        c = self.df_dec.df_out(c)
        c = c.view(b, t, self.df_dec.df_bins, self.df_dec.df_order * 2) + c0
        return c, h_out


def export_encoder(model_enc, model_dir, S):
    enc = StatefulEncoder(model_enc).eval()
    dummy_erb = torch.zeros(1, 1, S, 32)
    dummy_spec = torch.zeros(1, 2, S, 64)
    dummy_h = torch.zeros(ENC_NUM_LAYERS, 1, EMB_HIDDEN)
    out_path = os.path.join(model_dir, "enc.onnx")
    torch.onnx.export(
        enc,
        (dummy_erb, dummy_spec, dummy_h),
        out_path,
        input_names=["feat_erb", "feat_spec", "h_enc_in"],
        output_names=["e0", "e1", "e2", "e3", "emb", "c0", "lsnr", "h_enc_out"],
        dynamic_axes={
            "feat_erb": {2: "S"},
            "feat_spec": {2: "S"},
            "e0": {2: "S"},
            "e1": {2: "S"},
            "e2": {2: "S"},
            "e3": {2: "S"},
            "emb": {1: "S"},
            "c0": {2: "S"},
            "lsnr": {1: "S"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {out_path}")


def export_erb_decoder(model_erb_dec, model_dir, S):
    dec = StatefulErbDecoder(model_erb_dec).eval()
    dummy_emb = torch.zeros(1, S, EMB_OUT)
    dummy_e3 = torch.zeros(1, 16, S, 8)
    dummy_e2 = torch.zeros(1, 16, S, 8)
    dummy_e1 = torch.zeros(1, 16, S, 16)
    dummy_e0 = torch.zeros(1, 16, S, 32)
    dummy_h = torch.zeros(ERB_DEC_NUM_LAYERS, 1, EMB_HIDDEN)
    out_path = os.path.join(model_dir, "erb_dec.onnx")
    torch.onnx.export(
        dec,
        (dummy_emb, dummy_e3, dummy_e2, dummy_e1, dummy_e0, dummy_h),
        out_path,
        input_names=["emb", "e3", "e2", "e1", "e0", "h_erb_dec_in"],
        output_names=["m", "h_erb_dec_out"],
        dynamic_axes={
            "emb": {1: "S"},
            "e3": {2: "S"},
            "e2": {2: "S"},
            "e1": {2: "S"},
            "e0": {2: "S"},
            "m": {2: "S"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {out_path}")


def export_df_decoder(model_df_dec, model_dir, S):
    dec = StatefulDfDecoder(model_df_dec).eval()
    dummy_emb = torch.zeros(1, S, EMB_OUT)
    dummy_c0 = torch.zeros(1, 16, S, 64)
    dummy_h = torch.zeros(DF_DEC_NUM_LAYERS, 1, DF_HIDDEN)
    out_path = os.path.join(model_dir, "df_dec.onnx")
    torch.onnx.export(
        dec,
        (dummy_emb, dummy_c0, dummy_h),
        out_path,
        input_names=["emb", "c0", "h_df_dec_in"],
        output_names=["coefs", "h_df_dec_out"],
        dynamic_axes={
            "emb": {1: "S"},
            "c0": {2: "S"},
            "coefs": {1: "S"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {out_path}")


def main():
    ckpt_path = _ensure_hush_weights()
    config = get_config()
    model = DfNetSE(config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.model.load_state_dict(state)
    model.eval()

    out_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "livekit",
        "plugins",
        "hush",
        "models",
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    S = 3
    with torch.no_grad():
        export_encoder(model.model.enc, out_dir, S)
        export_erb_decoder(model.model.erb_dec, out_dir, S)
        export_df_decoder(model.model.df_dec, out_dir, S)

    print(f"\nExported 3 stateful ONNX models to {out_dir}")


if __name__ == "__main__":
    main()

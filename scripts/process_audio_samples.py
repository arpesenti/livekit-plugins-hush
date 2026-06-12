"""Process original WAV files through Hush noise suppression."""

import sys
import os
import wave
import struct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from livekit.plugins.hush._hush_model import (
    HushModel,
    HushSession,
    _CHUNK_SAMPLES,
    _SAMPLE_RATE,
)
from livekit.plugins.hush.noise_suppressor import _DEFAULT_MODEL_DIR


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        samples = samples / 255.0 * 2.0 - 1.0
    elif sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, framerate


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    int16 = np.clip(samples, -1.0, 1.0) * 32767.0
    int16 = int16.astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())


def main():
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "audio")
    originals_dir = os.path.join(docs_dir, "originals")

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "livekit", "plugins", "hush", "models"
    )

    print("Loading Hush model...")
    model = HushModel(model_path)
    session = HushSession(model)

    for fname in sorted(os.listdir(originals_dir)):
        if not fname.endswith(".wav"):
            continue

        in_path = os.path.join(originals_dir, fname)
        base = fname.replace(".wav", "")
        out_name = f"hush-{base}.wav"
        out_path = os.path.join(docs_dir, out_name)

        print(f"Processing {fname} -> {out_name} ...")
        audio, sr = read_wav(in_path)

        if sr != _SAMPLE_RATE:
            raise ValueError(f"Expected {_SAMPLE_RATE} Hz, got {sr} Hz for {fname}")

        # Process in chunks
        total = len(audio)
        output = np.empty(total, dtype=np.float32)
        pos = 0
        while pos < total:
            end = min(pos + _CHUNK_SAMPLES, total)
            chunk = audio[pos:end]
            if len(chunk) < _CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, _CHUNK_SAMPLES - len(chunk)))
            denoised = session.process_chunk(chunk)
            n_out = min(len(denoised), end - pos)
            output[pos : pos + n_out] = denoised[:n_out]
            pos += _CHUNK_SAMPLES

        write_wav(out_path, output, _SAMPLE_RATE)
        print(f"  Done: {out_path}")

    session.close()
    print("All files processed.")


if __name__ == "__main__":
    main()

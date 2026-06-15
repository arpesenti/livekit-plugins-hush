"""Process original WAV files through Hush noise suppression.

For each source file in ``docs/audio/originals/``, writes a single
denoised output to ``docs/audio/hush-{name}.wav`` using the
per-frame streaming Hush pipeline (one 10 ms frame at a time,
no resets — the same code path the LiveKit FrameProcessor uses in
production).

Dependencies: numpy. The Hush ONNX model bundle is loaded by the
plugin itself.
"""

import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from livekit.plugins.hush import HushNoiseSuppressor  # noqa: E402

SAMPLE_RATE = 16000
FRAME_SAMPLES = 160  # 10 ms at 16 kHz, the model's native frame size


def read_wav(path):
    with wave.open(path, "rb") as wf:
        nc, sw, sr, nf = (
            wf.getnchannels(),
            wf.getsampwidth(),
            wf.getframerate(),
            wf.getnframes(),
        )
        raw = wf.readframes(nf)
    if sw == 1:
        s = np.frombuffer(raw, np.uint8).astype(np.float32) / 255.0 * 2.0 - 1.0
    elif sw == 2:
        s = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        s = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    elif sw == 4:
        s = np.frombuffer(raw, np.float32)
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if nc > 1:
        s = s.reshape(-1, nc).mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {sr} Hz. Resample before passing to Hush."
        )
    return s.astype(np.float32)


def write_wav(path, samples, sr):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    i16 = np.clip(samples, -1.0, 1.0) * 32767.0
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.astype(np.int16).tobytes())


def _build_suppressor():
    """Construct a HushNoiseSuppressor with strength=1.0 (full model output)."""
    return HushNoiseSuppressor(strength=1.0)


def _process_frame(ns, audio_160):
    """Push a single 10 ms frame through the suppressor and return the output."""
    import livekit.rtc as rtc

    i16 = (np.clip(audio_160, -1.0, 1.0) * 32767.0).astype(np.int16)
    frame = rtc.AudioFrame(
        data=i16.tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=len(i16),
    )
    out = ns._process(frame)
    return np.frombuffer(out.data, dtype=np.int16).astype(np.float32) / 32768.0


def process(audio):
    """Process a full audio array with a single session, no resets."""
    ns = _build_suppressor()
    try:
        out = np.zeros_like(audio)
        pos = 0
        while pos + FRAME_SAMPLES <= len(audio):
            out[pos : pos + FRAME_SAMPLES] = _process_frame(
                ns, audio[pos : pos + FRAME_SAMPLES]
            )
            pos += FRAME_SAMPLES
        return out
    finally:
        ns._close()


def main():
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "audio")
    originals_dir = os.path.join(docs_dir, "originals")

    sources = sorted(f for f in os.listdir(originals_dir) if f.endswith(".wav"))
    if not sources:
        print(f"No .wav files found in {originals_dir}")
        return

    print(f"Processing {len(sources)} source file(s) with Hush denoiser...")
    for fname in sources:
        src_path = os.path.join(originals_dir, fname)
        audio = read_wav(src_path)
        print(f"  {fname}: {len(audio)} samples ({len(audio) / SAMPLE_RATE:.2f}s)")

        out = process(audio)
        write_wav(
            os.path.join(docs_dir, f"hush-{fname.replace('.wav', '')}.wav"),
            out,
            SAMPLE_RATE,
        )

    print(f"\nDone. Wrote {len(sources)} WAV files to {docs_dir}/")


if __name__ == "__main__":
    main()

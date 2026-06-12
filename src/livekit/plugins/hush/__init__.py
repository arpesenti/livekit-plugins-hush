from livekit.agents import Plugin
import logging

from .noise_suppressor import HushNoiseSuppressor
from ._hush_model import download_files

logger = logging.getLogger(__name__)


class HushPlugin(Plugin):
    def __init__(self):
        super().__init__(
            title="Hush",
            version="0.1.0",
            package="livekit-plugins-hush",
            logger=logger,
        )

    def download_files(self):
        download_files()


def noise_suppression(**kwargs) -> HushNoiseSuppressor:
    """Create a HushNoiseSuppressor instance.

    Pass to ``AudioInputOptions(noise_cancellation=hush.noise_suppression())``.

    Parameters
    ----------
    model_path : str, optional
        Path to the exported ONNX model file.
    chunk_frames : int
        Frames per inference chunk (default 32 = 320ms latency).
    atten_lim_db : float
        Maximum attenuation in dB (default 100.0).
    strength : float
        Wet/dry blend factor (default 0.5). 0.0 = bypass, 1.0 = full suppression.
    debug_logging : bool
        Log diagnostics every 10 chunks at DEBUG level.
    """
    return HushNoiseSuppressor(**kwargs)


Plugin.register_plugin(HushPlugin())

__all__ = ["HushNoiseSuppressor", "noise_suppression", "download_files"]

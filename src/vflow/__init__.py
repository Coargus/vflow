from __future__ import annotations

from . import _version
from .read_video import read_video
from .video import Video

__version__ = _version.__version__
del _version

__all__ = ["read_video", "Video"]

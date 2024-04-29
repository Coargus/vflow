"""Read Video Module."""

from __future__ import annotations

from pathlib import Path

from vflow.enum.video_format import VideoFormat
from vflow.video import Video


def read_video(video_path: str | Path) -> Video:
    """Read video from video_path.

    Args:
        video_path (str): Path to video file.

    Returns:
        Video: Video object.
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)

    if video_path.suffix == ".mp4":
        read_format = VideoFormat.MP4

    return Video(video_path, read_format=read_format)

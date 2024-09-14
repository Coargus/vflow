import uuid
from dataclasses import dataclass, field

from vflow.enum.video_format import VideoFormat


@dataclass
class VideoInfo:
    """Represents information about a video file."""

    format: VideoFormat
    frame_width: int
    frame_height: int
    original_frame_count: int
    video_id: uuid.UUID = field(default_factory=uuid.uuid4)
    video_path: str | None = None
    processed_fps: float | None = None
    processed_frame_count: int = 1
    original_fps: float | None = None

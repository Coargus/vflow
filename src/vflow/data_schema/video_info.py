import uuid
from dataclasses import dataclass, field

from vflow.enum.video_format import VideoFormat


@dataclass
class VideoInfo:
    """Represents information about a video file."""

    video_path: str
    format: VideoFormat
    frame_width: int
    frame_height: int
    original_fps: float
    original_frame_count: int
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    processed_fps: float | None = None
    processed_frame_count: int | None = None
    current_frame_index: int = 0
    video_ended: bool = False

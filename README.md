# vflow: Video Processing Library

vflow is a Python library for efficient video processing and frame extraction.

## Features

- Read video files (currently supports MP4)
- Extract video metadata
- Frame-by-frame processing
- Flexible frame extraction methods:
  - By desired FPS
  - By time interval

## Installation
```
pip install -e .
```

## Quick Start
```python
import vflow
# Read video
video = vflow.read_video("path/to/your/video.mp4")
# Get video info
print(video.get_video_info)

# Extract frames at 2 FPS
frames = video.get_all_frames_of_video(desired_fps=2)
# Or extract frames every 2 seconds
frames = video.get_all_frames_of_video(desired_interval_in_sec=2)
# Process frames one by one
while True:
    frame_img: np.ndarray = video.get_next_frame(
        return_format="ndarray",
        desired_interval_in_sec=1,
    )
    if frame_img is None:
        break  # No more frames or end of video
    print(f"Getting frame : {frame_count}")  # noqa: T201
```
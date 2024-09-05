import logging

import numpy as np

import vflow

# Configure logging
logging.basicConfig(level=logging.INFO)

# 1. Define video path
video_path = "/home/mc76728/repo/Coargus/multimodal-agentic-reasoning-system/tests/sample_data/sample-5s.mp4"

# 2. Define Video Object
video = vflow.read_video(video_path)

# 3. Process frames
frame_count = 0
while True:
    frame_img: np.ndarray = video.get_next_frame(
        return_format="ndarray",
        desired_interval_in_sec=1,
    )
    if frame_img is None:
        break  # No more frames or end of video
    print(f"Getting frame : {frame_count}")  # noqa: T201
    frame_count += 1

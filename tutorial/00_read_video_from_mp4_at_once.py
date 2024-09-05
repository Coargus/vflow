import vflow

# 1. Define video path
video_path = "/home/mc76728/repo/Coargus/multimodal-agentic-reasoning-system/tests/sample_data/sample-5s.mp4"

# 2. Define Video Object
# This creates a video object that we can use to extract information and frames
video = vflow.read_video(video_path)

# 3. Get and print video info before processing
# This shows the original metadata of the video
print("Video info before processing:")  # noqa: T201
print(video.get_video_info, "\n")  # noqa: T201
# 4. Get frames with desired FPS (Frames Per Second)
# This method extracts frames at a specific frame rate, in this case 2 FPS
fps_frames = video.get_all_frames_of_video(desired_fps=2)
print(f"Number of frames (FPS method): {len(fps_frames)}")  # noqa: T201
print("\nVideo info after processing:")  # noqa: T201
print(video.get_video_info)  # noqa: T201

print("--------------------------------------")
video = vflow.read_video(video_path)
# 5. Get and print video info before processing
# This shows the original metadata of the video
print("Video info before processing:")  # noqa: T201
print(video.get_video_info, "\n")  # noqa: T201
# 6. Get frames with desired interval
# This method extracts frames at regular time intervals, in this case every 2 seconds  # noqa: E501
interval_frames = video.get_all_frames_of_video(desired_interval_in_sec=2)
print(f"Number of frames (Interval method): {len(interval_frames)}")  # noqa: T201
print("\nVideo info after processing:")  # noqa: T201
print(video.get_video_info)  # noqa: T201

# Note: The number of frames extracted by the FPS and interval methods may differ # noqa: E501
# depending on the original video's duration and frame rate.

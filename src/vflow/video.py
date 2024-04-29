"""Vflow Video Object."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vflow.enum.video_format import VideoFormat


class Video:
    """Vflow's Video Object."""

    def __init__(
        self,
        video_path: str | Path,
        read_format: VideoFormat,
    ) -> None:
        """Video Frame Processor.

        Args:
            video_path (str): Path to video file.
            frame_duration_sec (int, optional): Frame duration in seconds. Defaults to 1.
            frame_scale (int | None, optional): Frame scale. Defaults to None.
        """
        self._video_path = video_path
        self._read_format = read_format
        self.current_frame_index = 0
        self.video_ended = False
        self.import_video(str(video_path))

    def _resize_frame_by_scale(
        self, frame_img: np.ndarray, frame_scale: int
    ) -> np.ndarray:
        """Resize frame image.

        Args:
            frame_img (np.ndarray): Frame image.
            frame_scale (int): Scale of frame.

        Returns:
            np.ndarray: Resized frame image.
        """
        return cv2.resize(
            frame_img,
            (
                int(self.original_video_width / frame_scale),
                int(self.original_video_height / frame_scale),
            ),
        )

    def import_video(self, video_path: str) -> None:
        """Read video from video_path.

        Args:
            video_path (str): Path to video file.
        """
        if self._read_format == VideoFormat.MP4:
            self._cap = cv2.VideoCapture(video_path)
            self.original_video_height = self._cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            self.original_video_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.original_video_fps = self._cap.get(cv2.CAP_PROP_FPS)
            self.original_frame_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_all_frames_of_video(
        self,
        return_format: str = "cv2",
        frame_scale: int | None = None,
        desired_fps: int | None = None,
        desired_interval: int | None = None,
    ) -> list:
        """Get video frames by frame_scale and second_per_frame.

        Args:
            return_format (str, optional): Return format. Defaults to "cv2".
                Options: [cv2, ndarray]
            frame_scale (int | None, optional): Frame scale. Defaults to None.
            desired_fps (int | None, optional): Desired FPS. Defaults to None.
            desired_interval (int | None, optional): Desired interval.
                Defaults to None.
        """
        all_frames = []

        if (
            self._read_format == VideoFormat.MP4
            and desired_fps is None
            and desired_interval is None
        ):
            msg = "Either desired_fps or desired_interval must be provided."
            raise ValueError(msg)

        if desired_fps is not None:
            frame_step = int(round(self.original_video_fps / desired_fps))
        if desired_interval is not None:
            frame_step = int(round(self.original_video_fps * desired_interval))

        for real_frame_idx in range(
            0, int(self.original_frame_count), int(frame_step)
        ):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_idx)
            ret, frame_img = self._cap.read()
            if not ret:
                break
            if frame_scale is not None:
                frame_img = self._resize_frame_by_scale(frame_img, frame_scale)
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            if return_format == "ndarray":
                frame_img = np.array(frame_img, dtype=np.uint8)
            all_frames.append(frame_img)
        self._cap.release()
        cv2.destroyAllWindows()
        return all_frames

    def get_next_frame(
        self,
        return_format: str = "ndarray",
        frame_scale: int | None = None,
        desired_fps: int | None = None,
        desired_interval: int | None = None,
    ) -> np.ndarray | None:
        """Get the next video frame based on frame step.

        Args:
            return_format (str, optional): Return format. Defaults to "ndarray".
                - [cv2, ndarray]
            frame_scale (int | None, optional): Frame scale. Defaults to None.
            desired_fps (int | None, optional): Desired FPS. Defaults to None.
            desired_interval (int | None, optional): Desired interval. Defaults to None.

        Returns:
            np.ndarray | None: The next frame as an ndarray, or None if no more frames are available or the video ended.
        """
        if (
            self._read_format == VideoFormat.MP4
            and desired_fps is None
            and desired_interval is None
        ):
            msg = "Either desired_fps or desired_interval must be provided."
            raise ValueError(msg)

        if self.video_ended:
            return None  # No more frames to process

        if desired_fps is not None:
            frame_step = int(round(self.original_video_fps / desired_fps))
        if desired_interval is not None:
            frame_step = int(round(self.original_video_fps * desired_interval))

        # Skip to the next frame based on frame_step
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

        ret, frame_img = self._cap.read()

        if not ret:
            self.video_ended = True
            return None  # No more frames or error occurred

        # Update the current frame index for the next call
        self.current_frame_index += frame_step

        if frame_scale is not None:
            frame_img = self._resize_frame_by_scale(frame_img, frame_scale)

        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        if return_format == "PIL.Image":
            # TODO: Convert to PIL.Image
            pass

        return frame_img

    def insert_annotation_to_current_frame(
        self, annotations: list[str]
    ) -> None:
        """Insert annotations to the current frame.

        Args:
            annotations (list[str]): List of annotations.
        """

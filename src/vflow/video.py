"""Vflow Video Object."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from vflow.data_schema.video_info import VideoInfo
from vflow.enum.video_format import VideoFormat


class Video:
    """vflow's Video Object."""

    def __init__(
        self,
        video_path: str | Path,
        read_format: VideoFormat,
    ) -> None:
        """Video Frame Processor.

        Args:
            video_path (str | Path): Path to video file.
            read_format (VideoFormat): Format to read the video in.
        """
        self._video_path = video_path
        self._read_format = read_format
        self.video_info = None
        self.import_video(str(video_path))

    def __str__(self) -> str:
        """Return a concise string representation of the Video object."""
        return str(self.video_info)

    def __repr__(self) -> str:
        """Return a detailed string representation of the Video object."""
        return repr(self.video_info)

    def import_video(self, video_path: str) -> None:
        """Read video from video_path.

        Args:
            video_path (str): Path to video file.
        """
        logging.info(f"Video format: {self._read_format}")
        if self._read_format == VideoFormat.MP4:
            self._cap = cv2.VideoCapture(video_path)
            self.video_info = VideoInfo(
                video_path=str(self._video_path),
                format=self._read_format,
                frame_width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                frame_height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                original_fps=self._cap.get(cv2.CAP_PROP_FPS),
                original_frame_count=int(
                    self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
                ),
            )

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
                int(self.video_info.frame_width / frame_scale),
                int(self.video_info.frame_height / frame_scale),
            ),
        )

    def get_all_frames_of_video(
        self,
        return_format: str = "cv2",
        frame_scale: int | None = None,
        desired_fps: int | None = None,
        desired_interval_in_sec: int | None = None,
    ) -> list:
        """Get video frames by frame_scale and second_per_frame.

        Args:
            return_format (str, optional): Return format. Defaults to "cv2".
                Options: [cv2, ndarray]
            frame_scale (int | None, optional): Frame scale. Defaults to None.
            desired_fps (int | None, optional): Desired FPS. Defaults to None.
            desired_interval_in_sec (int | None, optional): Interval between frames in seconds.
                If provided, frames will be extracted at this interval. Defaults to None.
        """  # noqa: E501
        all_frames = []

        if (
            self._read_format == VideoFormat.MP4
            and desired_fps is None
            and desired_interval_in_sec is None
        ):
            msg = "Either desired_fps or desired_interval_in_sec must be provided."
            raise ValueError(msg)

        if desired_fps is not None:
            frame_step = int(round(self.video_info.original_fps / desired_fps))
            self.video_info.processed_fps = desired_fps
        if desired_interval_in_sec is not None:
            frame_step = int(
                round(self.video_info.original_fps * desired_interval_in_sec)
            )
            self.video_info.processed_fps = round(
                1 / desired_interval_in_sec, 2
            )

        for real_frame_idx in range(
            0, int(self.video_info.original_frame_count), int(frame_step)
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
        self.video_info.processed_frame_count = len(all_frames)
        return all_frames

    def get_next_frame(
        self,
        return_format: str = "ndarray",
        frame_scale: int | None = None,
        desired_fps: int | None = None,
        desired_interval_in_sec: int | None = None,
    ) -> np.ndarray | None:
        """Get the next video frame based on frame step.

        Args:
            return_format (str, optional): Return format. Defaults to "ndarray".
                - [cv2, ndarray]
            frame_scale (int | None, optional): Frame scale. Defaults to None.
            desired_fps (int | None, optional): Desired FPS. Defaults to None.
            desired_interval_in_sec (int | None, optional): Desired interval. Defaults to None.

        Returns:
            np.ndarray | None: The next frame as an ndarray, or None if no more frames are available or the video ended.
        """
        if (
            self._read_format == VideoFormat.MP4
            and desired_fps is None
            and desired_interval_in_sec is None
        ):
            msg = "Either desired_fps or desired_interval_in_sec must be provided."
            raise ValueError(msg)

        if self.video_info.video_ended:
            return None  # No more frames to process

        if desired_fps is not None:
            frame_step = int(round(self.video_info.original_fps / desired_fps))
            self.video_info.processed_fps = desired_fps
        if desired_interval_in_sec is not None:
            frame_step = int(
                round(self.video_info.original_fps * desired_interval_in_sec)
            )
            self.video_info.processed_fps = round(
                1 / desired_interval_in_sec, 2
            )

        # Skip to the next frame based on frame_step
        self._cap.set(
            cv2.CAP_PROP_POS_FRAMES, self.video_info.current_frame_index
        )

        ret, frame_img = self._cap.read()

        if not ret:
            self.video_info.video_ended = True
            return None  # No more frames or error occurred

        # Update the current frame index for the next call
        self.video_info.current_frame_index += frame_step

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

    def get_video_info(self) -> VideoInfo:
        """Return the VideoInfo object containing video information."""
        return self.video_info

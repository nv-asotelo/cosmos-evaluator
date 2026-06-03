# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video helper utilities and writer."""

import base64
from contextlib import suppress
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, List, Sequence

import cv2
import numpy as np

# Must be a multiple of 3 for correct base64 chunk concatenation
_BASE64_CHUNK_BYTES = 3 * 1024 * 1024  # 3 MiB

logger = logging.getLogger(__name__)

_NVENC_AVAILABLE: bool | None = None

NVENC_H264_FFMPEG_ARGS: list[str] = [
    "-c:v",
    "h264_nvenc",
    "-pix_fmt",
    "yuv420p",
    "-preset",
    "p4",
    "-tune",
    "hq",
    "-rc",
    "vbr",
    "-cq",
    "23",
    "-b:v",
    "0",
    "-movflags",
    "+faststart",
]

OPENH264_FFMPEG_ARGS: list[str] = [
    "-c:v",
    "libopenh264",
    "-pix_fmt",
    "yuv420p",
    "-b:v",
    "2M",
    "-movflags",
    "+faststart",
]


def is_nvenc_available() -> bool:
    """Check if h264_nvenc is available by performing a test encode.

    Result is cached after the first call.
    """
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.1",
                "-vcodec",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        _NVENC_AVAILABLE = result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        _NVENC_AVAILABLE = False

    if _NVENC_AVAILABLE:
        logger.info("h264_nvenc hardware encoder detected")
    else:
        logger.info("h264_nvenc not available, will use libopenh264 CPU fallback")
    return _NVENC_AVAILABLE


def get_h264_ffmpeg_args(*, include_audio: bool = False) -> list[str]:
    """Return ffmpeg args for H.264 encoding, preferring NVENC with libopenh264 CPU fallback.

    Args:
        include_audio: If True, include ``-c:a aac`` for audio passthrough.
    """
    args = list(NVENC_H264_FFMPEG_ARGS if is_nvenc_available() else OPENH264_FFMPEG_ARGS)
    if include_audio:
        args.extend(["-c:a", "aac"])
    return args


def get_h264_encoder() -> str:
    """Return the best available H.264 encoder name (``h264_nvenc`` or ``libopenh264``)."""
    return "h264_nvenc" if is_nvenc_available() else "libopenh264"


def get_video_fps(video_path: str) -> float | None:
    """Get the FPS of a video.

    Args:
        video_path: The path to the video

    Returns:
        The FPS of the video or None if the video cannot be opened or the FPS cannot be determined
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        cap.release()  # Always release the capture object
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None:
        logger.warning(f"Failed to get FPS for video: {video_path}")
        return None

    fps = float(fps)
    if fps <= 0.0:
        logger.warning(f"Invalid FPS ({fps}) for video: {video_path}")
        return None

    return fps


def get_video_frame_count(video_path: Path) -> int | None:
    """Return total number of frames in a video if determinable.

    Args:
        video_path: The path to the video

    Returns:
        The number of frames in the video or None if the video cannot be opened or the frame count cannot be determined
    """
    try:
        if not video_path.exists():
            logger.error("Input video not found: {}".format(video_path))
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Failed to open video: {}".format(video_path))
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count <= 0:
            logger.warning("Unable to determine frame count for video: {}".format(video_path))
            return None

        return frame_count

    except Exception as e:
        logger.error("Error reading video frame count: {}".format(e))
        return None


def subsample_video_to_fps(source_path: str, out_path: str, target_fps: int = 4) -> bool:
    """Re-encode video to target_fps using cv2 (sample every step-th frame, then write).

    Reduces payload size for VLM by lowering frame rate. Returns True on success.
    """
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return False
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, round(source_fps / target_fps))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, float(target_fps), (width, height))
        if not out.isOpened():
            return False
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                out.write(frame)
            frame_idx += 1
        out.release()
        return frame_idx > 0
    finally:
        cap.release()


def encode_video_base64(video_path: str, target_fps: int = 4) -> tuple[str, str]:
    """Encode a local video file to base64 for VLM payloads.

    Subsamples to target_fps first to reduce payload size, then reads in 3 MiB
    chunks to avoid loading the entire file into memory.

    Returns:
        (mime_type, base64_string)
    """
    mime_type, _ = mimetypes.guess_type(video_path)
    mime_type = mime_type or "video/mp4"

    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    subsampled = subsample_video_to_fps(video_path, temp_path, target_fps=target_fps)
    path_to_encode = temp_path if subsampled else video_path

    try:
        chunks = []
        with open(path_to_encode, "rb") as f:
            while True:
                chunk = f.read(_BASE64_CHUNK_BYTES)
                if not chunk:
                    break
                chunks.append(base64.b64encode(chunk).decode("utf-8"))
        return mime_type, "".join(chunks)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def convert_video_ffmpeg(
    input_video_path: Path,
    ffmpeg_args: Sequence[str],
    hide_output: bool = False,
    output_video_path: Path | None = None,
) -> Path | None:
    """Convert a video using ffmpeg, optionally replacing the input file.

    Args:
        input_video_path: The path to the input video
        ffmpeg_args: The arguments to pass to ffmpeg
        hide_output: Whether to hide the output
        output_video_path: The path to the output video

    Returns:
        The path to the output video or None if the video cannot be converted
    """
    if not input_video_path.exists():
        error_msg = "Input video not found: {}".format(input_video_path)
        logger.error(error_msg)
        return None

    temp_dir = None
    try:
        if not output_video_path:
            temp_dir = tempfile.mkdtemp(dir=input_video_path.parent)
            temp_path = Path(temp_dir) / f"{input_video_path.stem}.compatible.mp4"
            replace_original = True
        else:
            temp_path = output_video_path
            replace_original = False

        cmd = ["ffmpeg", "-y", "-i", str(input_video_path), *ffmpeg_args, str(temp_path)]

        logger.info("Converting video with ffmpeg: {}".format(cmd))
        subprocess.run(cmd, capture_output=hide_output, text=True, check=True)

        if temp_path.exists():
            if replace_original:
                input_video_path.unlink()
                temp_path.rename(input_video_path)
                logger.info("Successfully converted video: {}".format(input_video_path))
                return input_video_path
            else:
                logger.info("Successfully converted video: {}".format(temp_path))
                return temp_path
        else:
            logger.error("ffmpeg conversion completed but output file not found")
            return None

    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg conversion failed: {}".format(e.stderr))
        return None
    except Exception as e:
        logger.error("Error during video conversion: {}".format(e))
        return None
    finally:
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temp directory: {}".format(temp_dir))
            except Exception as e:
                logger.error("Error during temp directory cleanup: {}".format(e))


def extract_keyframes(
    video_path: str,
    interval_seconds: float,
    jpeg_quality: int = 85,
    target_width: int | None = None,
    max_frames: int | None = None,
) -> List[bytes]:
    """
    Sample frames uniformly by time from the input video and return a list of JPEG bytes.

    Args:
        video_path: Path to the input video file
        interval_seconds: Sample a frame approximately every N seconds
        jpeg_quality: JPEG quality (1-100), defaults to 85
        target_width: If provided, resize frames by width while maintaining aspect ratio
        max_frames: If provided, stop after this many frames

    Returns:
        List of JPEG-encoded frames as bytes

    Raises:
        ValueError: If interval_seconds <= 0, video cannot be opened, or frame encoding fails
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be > 0 when provided")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0:
            logger.warning("Invalid FPS detected for {}, using fallback of 30 FPS".format(video_path))
            fps = 30.0
        frames_per_interval = max(round(fps * interval_seconds), 1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 1, 100))]

        def encode_frame(frame: np.ndarray) -> bytes:
            if target_width is not None and int(target_width) > 0:
                orig_h, orig_w = int(frame.shape[0]), int(frame.shape[1])
                if orig_w > 0:
                    new_w = int(target_width)
                    new_h = max(1, round(orig_h * (new_w / float(orig_w))))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, jpeg_params)
            if not ok:
                raise ValueError("Failed to encode frame to JPEG")
            return buf.tobytes()

        if max_frames is not None and total_frames > 0:
            candidate_indices = list(range(0, total_frames, frames_per_interval)) or [0]
            if len(candidate_indices) > max_frames:
                if max_frames == 1:
                    selected_indices = [candidate_indices[0]]
                else:
                    selected_indices = [
                        candidate_indices[round(i * (len(candidate_indices) - 1) / (max_frames - 1))]
                        for i in range(max_frames)
                    ]
            else:
                selected_indices = candidate_indices

            collected = []
            for frame_index in selected_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = cap.read()
                if success:
                    collected.append(encode_frame(frame))
            return collected

        frame_index = 0
        collected: List[bytes] = []
        success, frame = cap.read()
        while success:
            if frame_index % frames_per_interval == 0:
                collected.append(encode_frame(frame))
                if max_frames is not None and len(collected) >= max_frames:
                    break
            frame_index += 1
            success, frame = cap.read()

        return collected
    finally:
        cap.release()


def jpeg_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    """Convert JPEG bytes to a data URL suitable for OpenAI image_url content.

    Args:
        jpeg_bytes: The JPEG bytes to convert

    Returns:
        The data URL
    """
    import base64

    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class VideoWriter:
    """
    Helper for writing videos with optional per-frame preprocessing and
    automatic ffmpeg post-conversion for compatibility.

    Accepts RGB frames and handles color conversion and resizing internally.
    """

    def __init__(
        self,
        output_path: Path,
        fps: float | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        preprocessor: Callable[[np.ndarray, int], np.ndarray] | None = None,
        ffmpeg_args: Sequence[str] | None = None,
    ) -> None:
        """Initialize the VideoWriter

        Args:
            output_path: The path to the output video
            fps: The FPS of the video
            frame_width: The width of the video
            frame_height: The height of the video
            preprocessor: The preprocessor to apply to the video
            ffmpeg_args: The arguments to pass to ffmpeg
        """
        self.output_path = output_path
        self.fps = fps if fps is not None else 10.0
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.preprocessor = preprocessor
        self.ffmpeg_args = list(ffmpeg_args) if ffmpeg_args is not None else get_h264_ffmpeg_args()

        self.video_writer: cv2.VideoWriter | None = None
        self._frames_written: int = 0
        # ensure directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_writer(self) -> bool:
        try:
            # Try codecs in order of preference
            codecs_to_try = ["mp4v", "avc1", "XVID", "MJPG"]
            for codec in codecs_to_try:
                writer = None
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(
                        str(self.output_path), fourcc, float(self.fps), (int(self.frame_width), int(self.frame_height))
                    )
                    if writer.isOpened():
                        self.video_writer = writer
                        return True
                    if writer:
                        writer.release()
                except Exception as e:
                    logger.debug("Writer init failed for codec {}: {}".format(codec, e))
                    if writer:
                        with suppress(Exception):
                            writer.release()
            logger.warning("Failed to initialize video writer with any codec")
            return False
        except Exception as e:
            logger.warning("Video writer init failed: {}".format(e))
            return False

    def open(self) -> bool:
        """Eagerly initialize the underlying cv2 writer.

        Returns:
            True if the writer was initialized successfully
        """
        return self._initialize_writer()

    def write_frame(self, frame_rgb: np.ndarray, frame_idx: int) -> bool:
        """Write a frame (RGB) to the video, applying optional preprocessing.

        Args:
            frame_rgb: The frame to write
            frame_idx: The index of the frame

        Returns:
            True if the frame was written successfully
        """
        try:
            if frame_rgb is None:
                return False

            # Initialize dimensions on first write if not provided
            if self.frame_width is None or self.frame_height is None:
                self.frame_height, self.frame_width = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])

            # Lazy init writer
            if self.video_writer is None:
                if not self._initialize_writer():
                    return False

            # Ensure size matches
            if frame_rgb.shape[:2] != (self.frame_height, self.frame_width):
                frame_rgb = cv2.resize(frame_rgb, (self.frame_width, self.frame_height))

            # Apply preprocessor (e.g., legend), if any
            if self.preprocessor is not None:
                try:
                    frame_rgb = self.preprocessor(frame_rgb, frame_idx)
                except Exception as e:
                    logger.error("Frame preprocessing failed at {}: {}".format(frame_idx, e))

            # Convert to BGR and uint8
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if frame_bgr.dtype != np.uint8:
                frame_bgr = frame_bgr.astype(np.uint8)

            self.video_writer.write(frame_bgr)
            self._frames_written += 1
            return True
        except Exception as e:
            logger.error("Error writing frame {}: {}".format(frame_idx, e))
            return False

    def release(self) -> None:
        """Release writer and convert video for compatibility if frames were written."""
        if self.video_writer:
            try:
                self.video_writer.release()
            except Exception:
                pass
            if self.output_path and self._frames_written > 0:
                maybe_output = convert_video_ffmpeg(
                    input_video_path=self.output_path,
                    ffmpeg_args=self.ffmpeg_args,
                    hide_output=True,
                )
                if maybe_output:
                    self.output_path = maybe_output

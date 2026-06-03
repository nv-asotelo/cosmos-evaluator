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

"""Unit tests for image module."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from checks.utils.overlay_video_helper import OverlayVideoHelper
import checks.utils.video as video


class TestIsNvencAvailable(unittest.TestCase):
    """Tests for is_nvenc_available function."""

    def setUp(self):
        video._NVENC_AVAILABLE = None

    def tearDown(self):
        video._NVENC_AVAILABLE = None

    @patch("checks.utils.video.subprocess.run")
    def test_returns_true_when_nvenc_works(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(video.is_nvenc_available())

    @patch("checks.utils.video.subprocess.run")
    def test_returns_false_when_nvenc_fails(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="No capable devices found")
        self.assertFalse(video.is_nvenc_available())

    @patch("checks.utils.video.subprocess.run")
    def test_caches_result(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        video.is_nvenc_available()
        video.is_nvenc_available()
        mock_run.assert_called_once()

    @patch("checks.utils.video.subprocess.run")
    def test_timeout_returns_false(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5)
        self.assertFalse(video.is_nvenc_available())

    @patch("checks.utils.video.subprocess.run")
    def test_oserror_returns_false(self, mock_run):
        mock_run.side_effect = OSError("ffmpeg not found")
        self.assertFalse(video.is_nvenc_available())


class TestGetH264FfmpegArgs(unittest.TestCase):
    """Tests for get_h264_ffmpeg_args and get_h264_encoder functions."""

    def tearDown(self):
        video._NVENC_AVAILABLE = None

    @patch("checks.utils.video.is_nvenc_available", return_value=True)
    def test_returns_nvenc_args_when_available(self, _):
        args = video.get_h264_ffmpeg_args()
        self.assertIn("h264_nvenc", args)
        self.assertNotIn("libopenh264", args)

    @patch("checks.utils.video.is_nvenc_available", return_value=False)
    def test_returns_openh264_args_when_nvenc_unavailable(self, _):
        args = video.get_h264_ffmpeg_args()
        self.assertIn("libopenh264", args)
        self.assertNotIn("h264_nvenc", args)

    @patch("checks.utils.video.is_nvenc_available", return_value=True)
    def test_include_audio_adds_aac(self, _):
        args = video.get_h264_ffmpeg_args(include_audio=True)
        self.assertIn("-c:a", args)
        self.assertIn("aac", args)

    @patch("checks.utils.video.is_nvenc_available", return_value=True)
    def test_no_audio_by_default(self, _):
        args = video.get_h264_ffmpeg_args()
        self.assertNotIn("-c:a", args)

    @patch("checks.utils.video.is_nvenc_available", return_value=True)
    def test_get_h264_encoder_returns_nvenc(self, _):
        self.assertEqual(video.get_h264_encoder(), "h264_nvenc")

    @patch("checks.utils.video.is_nvenc_available", return_value=False)
    def test_get_h264_encoder_returns_openh264(self, _):
        self.assertEqual(video.get_h264_encoder(), "libopenh264")


class TestConvertVideoFFmpeg(unittest.TestCase):
    """Test cases for convert_video_ffmpeg function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_video = self.test_dir / "test_input.mp4"
        self.output_video = self.test_dir / "test_output.mp4"

        # Create a dummy input video file
        self.input_video.write_text("dummy video content")

        # Sample ffmpeg arguments
        self.sample_ffmpeg_args = ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p4"]

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory and all contents
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_convert_video_with_explicit_output_path_success(self):
        """Test successful video conversion with explicit output path."""
        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create output file to simulate ffmpeg success
            self.output_video.write_text("converted video content")

            result = video.convert_video_ffmpeg(
                input_video_path=self.input_video,
                ffmpeg_args=self.sample_ffmpeg_args,
                output_video_path=self.output_video,
            )

            # Verify result
            self.assertEqual(result, self.output_video)

            # Verify subprocess was called with correct arguments
            expected_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.input_video),
                "-c:v",
                "h264_nvenc",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p4",
                str(self.output_video),
            ]
            mock_run.assert_called_once_with(expected_cmd, capture_output=False, text=True, check=True)

    @patch("tempfile.mkdtemp")
    def test_convert_video_replace_original_success(self, mock_mkdtemp):
        """Test successful video conversion that replaces original file."""
        # Setup mock temp directory
        temp_dir = self.test_dir / "temp_conversion"
        temp_dir.mkdir()
        mock_mkdtemp.return_value = str(temp_dir)

        temp_output = temp_dir / "test_input.compatible.mp4"

        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create temp output file to simulate ffmpeg success
            temp_output.write_text("converted video content")

            result = video.convert_video_ffmpeg(
                input_video_path=self.input_video, ffmpeg_args=self.sample_ffmpeg_args, output_video_path=None
            )

            # Verify result - should return the original path (after replacement)
            self.assertEqual(result, self.input_video)

            # Verify temp directory was created in parent of input
            mock_mkdtemp.assert_called_once_with(dir=self.input_video.parent)

            # Verify subprocess was called with correct temp path
            expected_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.input_video),
                "-c:v",
                "h264_nvenc",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p4",
                str(temp_output),
            ]
            mock_run.assert_called_once_with(expected_cmd, capture_output=False, text=True, check=True)

    def test_convert_video_input_not_exists(self):
        """Test conversion with non-existent input file."""
        non_existent_input = self.test_dir / "non_existent.mp4"

        with patch("checks.utils.video.logger") as mock_logger:
            result = video.convert_video_ffmpeg(
                input_video_path=non_existent_input,
                ffmpeg_args=self.sample_ffmpeg_args,
                output_video_path=self.output_video,
            )

            # Verify function returns None
            self.assertIsNone(result)

            # Verify error was logged
            mock_logger.error.assert_called_once_with("Input video not found: {}".format(non_existent_input))

    def test_convert_video_subprocess_error(self):
        """Test conversion when subprocess fails."""
        with patch("subprocess.run") as mock_run:
            # Mock subprocess failure
            error = subprocess.CalledProcessError(returncode=1, cmd="ffmpeg", stderr="ffmpeg error message")
            mock_run.side_effect = error

            with patch("checks.utils.video.logger") as mock_logger:
                result = video.convert_video_ffmpeg(
                    input_video_path=self.input_video,
                    ffmpeg_args=self.sample_ffmpeg_args,
                    output_video_path=self.output_video,
                )

                # Verify function returns None
                self.assertIsNone(result)

                # Verify error was logged
                mock_logger.error.assert_called_once_with("ffmpeg conversion failed: ffmpeg error message")

    def test_convert_video_output_file_not_created(self):
        """Test conversion when ffmpeg succeeds but output file isn't created."""
        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()
            # Don't create output file to simulate this edge case

            with patch("checks.utils.video.logger") as mock_logger:
                result = video.convert_video_ffmpeg(
                    input_video_path=self.input_video,
                    ffmpeg_args=self.sample_ffmpeg_args,
                    output_video_path=self.output_video,
                )

                # Verify function returns None
                self.assertIsNone(result)

                # Verify error was logged
                mock_logger.error.assert_called_once_with("ffmpeg conversion completed but output file not found")

    def test_convert_video_general_exception(self):
        """Test conversion when an unexpected exception occurs."""
        with patch("subprocess.run") as mock_run:
            # Mock unexpected exception
            mock_run.side_effect = OSError("Unexpected OS error")

            with patch("checks.utils.video.logger") as mock_logger:
                result = video.convert_video_ffmpeg(
                    input_video_path=self.input_video,
                    ffmpeg_args=self.sample_ffmpeg_args,
                    output_video_path=self.output_video,
                )

                # Verify function returns None
                self.assertIsNone(result)

                # Verify error was logged
                mock_logger.error.assert_called_once_with("Error during video conversion: Unexpected OS error")

    def test_convert_video_empty_ffmpeg_args(self):
        """Test conversion with empty ffmpeg arguments."""
        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create output file to simulate ffmpeg success
            self.output_video.write_text("converted video content")

            result = video.convert_video_ffmpeg(
                input_video_path=self.input_video,
                ffmpeg_args=[],  # Empty args
                output_video_path=self.output_video,
            )

            # Verify result
            self.assertEqual(result, self.output_video)

            # Verify subprocess was called with minimal arguments
            expected_cmd = ["ffmpeg", "-y", "-i", str(self.input_video), str(self.output_video)]
            mock_run.assert_called_once_with(expected_cmd, capture_output=False, text=True, check=True)

    def test_convert_video_complex_ffmpeg_args(self):
        """Test conversion with complex ffmpeg arguments."""
        complex_args = [
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
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
        ]

        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create output file to simulate ffmpeg success
            self.output_video.write_text("converted video content")

            result = video.convert_video_ffmpeg(
                input_video_path=self.input_video, ffmpeg_args=complex_args, output_video_path=self.output_video
            )

            # Verify result
            self.assertEqual(result, self.output_video)

            # Verify subprocess was called with all arguments
            expected_cmd = ["ffmpeg", "-y", "-i", str(self.input_video), *complex_args, str(self.output_video)]
            mock_run.assert_called_once_with(expected_cmd, capture_output=False, text=True, check=True)

    @patch("checks.utils.video.logger")
    def test_convert_video_logging_success_with_output_path(self, mock_logger):
        """Test that success logging works correctly with explicit output path."""
        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create output file to simulate ffmpeg success
            self.output_video.write_text("converted video content")

            video.convert_video_ffmpeg(
                input_video_path=self.input_video,
                ffmpeg_args=self.sample_ffmpeg_args,
                output_video_path=self.output_video,
            )

            # Verify logging calls
            expected_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.input_video),
                "-c:v",
                "h264_nvenc",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p4",
                str(self.output_video),
            ]
            expected_calls = [
                call("Converting video with ffmpeg: {}".format(expected_cmd)),
                call("Successfully converted video: {}".format(self.output_video)),
            ]
            mock_logger.info.assert_has_calls(expected_calls)

    @patch("tempfile.mkdtemp")
    @patch("checks.utils.video.logger")
    def test_convert_video_logging_success_replace_original(self, mock_logger, mock_mkdtemp):
        """Test that success logging works correctly when replacing original."""
        # Setup mock temp directory
        temp_dir = self.test_dir / "temp_conversion"
        temp_dir.mkdir()
        mock_mkdtemp.return_value = str(temp_dir)

        temp_output = temp_dir / "test_input.compatible.mp4"

        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create temp output file to simulate ffmpeg success
            temp_output.write_text("converted video content")

            video.convert_video_ffmpeg(
                input_video_path=self.input_video, ffmpeg_args=self.sample_ffmpeg_args, output_video_path=None
            )

            # Verify logging calls
            expected_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.input_video),
                "-c:v",
                "h264_nvenc",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p4",
                str(temp_output),
            ]
            expected_calls = [
                call("Converting video with ffmpeg: {}".format(expected_cmd)),
                call("Successfully converted video: {}".format(self.input_video)),
            ]
            mock_logger.info.assert_has_calls(expected_calls)

    def test_convert_video_hide_output_true(self):
        """Test that hide_output=True sets capture_output=True."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            self.output_video.write_text("converted video content")

            video.convert_video_ffmpeg(
                input_video_path=self.input_video,
                ffmpeg_args=self.sample_ffmpeg_args,
                output_video_path=self.output_video,
                hide_output=True,
            )

            # Verify capture_output=True when hide_output=True
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            self.assertTrue(kwargs["capture_output"])


class TestGetVideoFrameCount(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.video_path = self.test_dir / "video.mp4"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_missing_file_returns_none_and_logs(self):
        with patch("checks.utils.video.logger") as mock_logger:
            count = video.get_video_frame_count(self.video_path)
            self.assertIsNone(count)
            mock_logger.error.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_cannot_open_returns_none(self, mock_cap_cls):
        self.video_path.write_text("dummy")
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cap_cls.return_value = cap
        with patch("checks.utils.video.logger") as mock_logger:
            count = video.get_video_frame_count(self.video_path)
            self.assertIsNone(count)
            mock_logger.error.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_success_returns_integer_frame_count(self, mock_cap_cls):
        self.video_path.write_text("dummy")
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 123
        mock_cap_cls.return_value = cap
        count = video.get_video_frame_count(self.video_path)
        self.assertEqual(count, 123)
        cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_non_positive_count_returns_none_with_warning(self, mock_cap_cls):
        self.video_path.write_text("dummy")
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0
        mock_cap_cls.return_value = cap
        with patch("checks.utils.video.logger") as mock_logger:
            count = video.get_video_frame_count(self.video_path)
            self.assertIsNone(count)
            mock_logger.warning.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_exception_returns_none_and_logs(self, mock_cap_cls):
        self.video_path.write_text("dummy")
        mock_cap_cls.side_effect = RuntimeError("OpenCV error")
        with patch("checks.utils.video.logger") as mock_logger:
            count = video.get_video_frame_count(self.video_path)
            self.assertIsNone(count)
            mock_logger.error.assert_called()


class TestGetVideoFPS(unittest.TestCase):
    @patch("cv2.VideoCapture")
    def test_success_returns_fps_and_releases(self, mock_cap_cls):
        cap = MagicMock()
        cap.get.return_value = 29.97
        mock_cap_cls.return_value = cap

        fps = video.get_video_fps("/path/to/video.mp4")

        self.assertEqual(fps, 29.97)
        mock_cap_cls.assert_called_once_with("/path/to/video.mp4")
        cap.get.assert_called_once_with(video.cv2.CAP_PROP_FPS)
        cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_returns_none_when_fps_zero(self, mock_cap_cls):
        """Test that get_video_fps returns None when FPS is zero or invalid."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        mock_cap_cls.return_value = cap

        fps = video.get_video_fps("/v.mp4")
        self.assertIsNone(fps)
        cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_failed_open_returns_none_and_logs(self, mock_cap_cls):
        """Test that get_video_fps returns None and logs error when video cannot be opened."""
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cap_cls.return_value = cap

        with patch("checks.utils.video.logger") as mock_logger:
            fps = video.get_video_fps("/bad/video.mp4")
            self.assertIsNone(fps)
            mock_logger.error.assert_called_once()

        cap.release.assert_called_once()
        mock_cap_cls.assert_called_once_with("/bad/video.mp4")


class TestExtractKeyframes(unittest.TestCase):
    @patch("cv2.VideoCapture")
    def test_max_frames_samples_across_timeline(self, mock_cap_cls):
        cap = MagicMock()
        cap.isOpened.return_value = True

        def get_prop(prop):
            if prop == video.cv2.CAP_PROP_FPS:
                return 10.0
            if prop == video.cv2.CAP_PROP_FRAME_COUNT:
                return 100
            return 0

        cap.get.side_effect = get_prop
        cap.read.return_value = (True, np.full((36, 64, 3), 127, dtype=np.uint8))
        mock_cap_cls.return_value = cap

        frames = video.extract_keyframes("/path/to/video.mp4", interval_seconds=1.0, max_frames=5)

        self.assertEqual(len(frames), 5)
        cap.set.assert_has_calls(
            [
                call(video.cv2.CAP_PROP_POS_FRAMES, 0),
                call(video.cv2.CAP_PROP_POS_FRAMES, 20),
                call(video.cv2.CAP_PROP_POS_FRAMES, 40),
                call(video.cv2.CAP_PROP_POS_FRAMES, 70),
                call(video.cv2.CAP_PROP_POS_FRAMES, 90),
            ]
        )
        cap.release.assert_called_once()


class TestVideoOverlayHelper(unittest.TestCase):
    """Tests for VideoOverlayHelper class."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "out"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    def test_initialize_video_writer_success(self, mock_cap_cls, mock_writer_cls):
        cap = MagicMock()
        cap.isOpened.return_value = True
        mock_cap_cls.return_value = cap

        writer = MagicMock()
        writer.isOpened.return_value = True
        mock_writer_cls.return_value = writer

        helper = OverlayVideoHelper(clip_id="clip1", output_dir=str(self.output_dir))
        ok = helper.initialize_video_writer(
            video_path="/path/to/input.mp4",
            naming_suffix="renamed_suffix",
            target_width=320,
            target_height=240,
            fps=12.5,
        )

        self.assertTrue(ok)
        self.assertEqual(helper.frame_width, 320)
        self.assertEqual(helper.frame_height, 240)
        self.assertEqual(helper.fps, 12.5)
        self.assertTrue(str(helper.video_output_path).endswith("clip1.renamed_suffix.mp4"))
        mock_cap_cls.assert_called_once_with("/path/to/input.mp4")
        mock_writer_cls.assert_called()  # at least one codec attempt

    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    def test_initialize_video_writer_fails_opening_video(self, mock_cap_cls, mock_writer_cls):
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cap_cls.return_value = cap

        helper = OverlayVideoHelper(clip_id="clip1", output_dir=str(self.output_dir))
        ok = helper.initialize_video_writer(video_path="/bad/input.mp4")
        self.assertFalse(ok)
        mock_writer_cls.assert_not_called()

    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    def test_initialize_video_writer_no_codec_works(self, mock_cap_cls, mock_writer_cls):
        cap = MagicMock()
        cap.isOpened.return_value = True
        mock_cap_cls.return_value = cap

        # Writer that never opens
        writer = MagicMock()
        writer.isOpened.return_value = False
        mock_writer_cls.return_value = writer

        helper = OverlayVideoHelper(clip_id="clip1", output_dir=str(self.output_dir))
        ok = helper.initialize_video_writer(video_path="/path/to/input.mp4")
        self.assertFalse(ok)

    def test_get_original_frame_success(self):
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        helper.frame_width = 64
        helper.frame_height = 48

        # Create a blue-ish BGR image 20x30
        bgr = np.zeros((30, 20, 3), dtype=np.uint8)
        bgr[..., 0] = 255

        cap = MagicMock()
        cap.read.return_value = (True, bgr)
        helper.original_video_cap = cap

        rgb = helper.get_original_frame(frame_idx=5)
        self.assertEqual(rgb.shape, (48, 64, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_get_original_frame_fallback_gray(self):
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        helper.frame_width = 32
        helper.frame_height = 24

        cap = MagicMock()
        cap.read.return_value = (False, None)
        helper.original_video_cap = cap

        img = helper.get_original_frame(frame_idx=0)
        self.assertEqual(img.shape, (24, 32, 3))
        self.assertTrue(np.all(img == 128))

    def test_write_frame_with_resize_and_dtype_cast(self):
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        helper.frame_width = 50
        helper.frame_height = 40

        # Float32 RGB frame with wrong size
        frame = np.random.rand(10, 20, 3).astype(np.float32) * 255.0

        # Mock writer
        writer = MagicMock()
        writer.isOpened.return_value = True
        helper.video_writer = writer

        ok = helper.write_frame(frame=frame, frame_idx=1)
        self.assertTrue(ok)
        # When using our VideoWriter, write() would be delegated internally; here we mocked
        # a raw cv2 writer so ensure write() was called
        writer.write.assert_called_once()
        written = writer.write.call_args[0][0]
        self.assertEqual(written.shape, (40, 50, 3))
        self.assertEqual(written.dtype, np.uint8)

    def test_write_frame_when_writer_missing(self):
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        helper.frame_width = 10
        helper.frame_height = 10
        helper.video_writer = None

        with patch.object(helper, "logger") as mock_logger:
            ok = helper.write_frame(frame=np.zeros((10, 10, 3), dtype=np.uint8), frame_idx=2)
            self.assertFalse(ok)
            mock_logger.warning.assert_called()

    @patch("cv2.imwrite")
    def test_save_frame_image_success(self, mock_imwrite):
        mock_imwrite.return_value = True
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        ok = helper.save_frame_image(frame=frame, frame_idx=3)
        self.assertTrue(ok)
        expected_path = self.output_dir / "overlay_frame_0003.png"
        args, _ = mock_imwrite.call_args
        self.assertEqual(Path(args[0]), expected_path)

    @patch("cv2.imwrite")
    def test_save_frame_image_exception(self, mock_imwrite):
        mock_imwrite.side_effect = RuntimeError("disk error")
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        with patch.object(helper, "logger") as mock_logger:
            ok = helper.save_frame_image(frame=np.zeros((4, 4, 3), dtype=np.uint8), frame_idx=1)
            self.assertFalse(ok)
            mock_logger.error.assert_called()

    def test_release_converts_and_releases(self):
        helper = OverlayVideoHelper(clip_id="clipX", output_dir=str(self.output_dir))
        helper.video_output_path = self.output_dir / "clipX.overlay.mp4"

        writer = MagicMock()
        helper.video_writer = writer

        cap = MagicMock()
        helper.original_video_cap = cap

        with patch("checks.utils.overlay_video_helper.convert_video_ffmpeg", return_value=None) as mock_convert:
            helper.release()

        writer.release.assert_called_once()
        cap.release.assert_called_once()
        # We no longer update video_output_path in the base helper (VideoWriter handles it),
        # but conversion is still invoked for compatibility
        self.assertEqual(helper.video_output_path, self.output_dir / "clipX.overlay.mp4")
        mock_convert.assert_called_once()

    def test_release_handles_no_convert_result(self):
        helper = OverlayVideoHelper(clip_id="clipY", output_dir=str(self.output_dir))
        original = self.output_dir / "clipY.overlay.mp4"
        helper.video_output_path = original

        writer = MagicMock()
        helper.video_writer = writer

        with patch("checks.utils.overlay_video_helper.convert_video_ffmpeg", return_value=None) as mock_convert:
            helper.release()

        writer.release.assert_called_once()
        self.assertEqual(helper.video_output_path, original)
        mock_convert.assert_called_once()

    def test_add_legend_draws_expected_elements(self):
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        helper.frame_width = 200
        helper.frame_height = 120

        vis = np.zeros((120, 200, 3), dtype=np.uint8)
        legend = ["Item A", "Item B", "Item C"]

        # Mock OpenCV text sizing and drawing to assert calls/positions
        with (
            patch("cv2.getTextSize") as mock_get_text_size,
            patch("cv2.rectangle") as mock_rectangle,
            patch("cv2.addWeighted") as mock_add_weighted,
            patch("cv2.putText") as mock_put_text,
        ):
            # Return fixed size for any text queried
            mock_get_text_size.return_value = ((100, 12), 3)

            # Make addWeighted return a sentinel object to verify the return value
            blended_sentinel = object()
            mock_add_weighted.return_value = blended_sentinel

            result = helper.add_legend(vis.copy(), frame_idx=7, legend_items=legend)

            # Validate addWeighted called with overlay copy and original image
            self.assertTrue(mock_add_weighted.called)
            aw_args, aw_kwargs = mock_add_weighted.call_args
            overlay_param = aw_args[0]
            base_param = aw_args[2]
            self.assertIsInstance(overlay_param, np.ndarray)
            self.assertIsNot(overlay_param, vis)
            self.assertEqual(overlay_param.shape, vis.shape)
            self.assertIs(base_param, mock_add_weighted.call_args[0][2])
            self.assertEqual(aw_args[1], 0.65)
            self.assertEqual(aw_args[3], 0.35)
            self.assertEqual(aw_args[4], 0)

            # Rectangle should be drawn around the legend area with expected coords
            # Using fixed text sizes: frame_h=12, margin=14, pad=12, legend_height=3*(12+4)=48
            expected_tl = (14 - 12, (12 + 44) - 12 - 12)  # (2, 32)
            expected_br = (14 + max(100, 100) + 12, (12 + 44) + 48 + 12)  # (126, 116)
            mock_rectangle.assert_called_once_with(overlay_param, expected_tl, expected_br, (24, 28, 32), -1)

            # putText should be called for frame text + each legend item at expected positions
            expected_calls = []
            # Frame text at (14, 56)
            expected_calls.append(
                call(blended_sentinel, "Frame: 7", (14, 56), video.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 240, 230), 2)
            )
            # Legend items starting at y = 56 + 12 + 10 = 78, step = 16
            legend_start_y = 56 + 12 + 10
            for i, item in enumerate(legend):
                expected_calls.append(
                    call(
                        blended_sentinel,
                        item,
                        (14, legend_start_y + i * (12 + 4)),
                        video.cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (250, 240, 230),
                        2,
                    )
                )

            # putText is called in order; verify subset and order
            mock_put_text.assert_has_calls(expected_calls, any_order=False)

            # The returned image is whatever addWeighted returned (then drawn on)
            self.assertIs(result, blended_sentinel)


if __name__ == "__main__":
    unittest.main()

from typing import Iterable

import cv2

from src.video.pipelines import (
    DELTA_VIDEO_DECODING_PIPELINE,
    DELTA_VIDEO_ENCODING_PIPELINE,
    LINEAR_VIDEO_DECODING_PIPELINE,
    LINEAR_VIDEO_ENCODING_PIPELINE,
    PREDICTIVE_DELTA_VIDEO_DECODING_PIPELINE, PREDICTIVE_DELTA_VIDEO_ENCODING_PIPELINE,
)


def _frames_generator(video_capture, max_frames=None):
    count = 0
    while video_capture.isOpened():
        frames = video_capture.read()[1]
        count += 1
        if frames is None or (max_frames and count > max_frames):
            break
        # Convert frame to YCbCr
        frames = cv2.cvtColor(frames, cv2.COLOR_RGB2YCrCb)

        yield frames


def video_reader(file_path: str, max_frames=None):
    return _frames_generator(cv2.VideoCapture(file_path), max_frames)


def encode_video(data: Iterable, compression_level: int = 1) -> bytes:
    result = LINEAR_VIDEO_ENCODING_PIPELINE.apply(
        data,
        run_settings={
            "compression_level": compression_level,
        },
    )[0]

    return result


def encode_video_delta(
        data: Iterable, compression_level: int = 1, reset_every: int = 10
) -> bytes:
    result = DELTA_VIDEO_ENCODING_PIPELINE.apply(
        data,
        run_settings={
            "reset_every": reset_every,
            "compression_level": compression_level,
        },
    )[0]

    return result


def encode_video_delta_predictive(
        data: Iterable, compression_level: int = 1, reset_every: int = 10
) -> bytes:
    result = PREDICTIVE_DELTA_VIDEO_ENCODING_PIPELINE.apply(
        data,
        run_settings={
            "reset_every": reset_every,
            "compression_level": compression_level,
        },
    )[0]

    return result


def decode_video(data: bytes) -> Iterable:
    return LINEAR_VIDEO_DECODING_PIPELINE.apply(data)[0]


def decode_video_delta(data: bytes) -> Iterable:
    return DELTA_VIDEO_DECODING_PIPELINE.apply(data)[0]


def decode_video_delta_predictive(data: bytes) -> Iterable:
    return PREDICTIVE_DELTA_VIDEO_DECODING_PIPELINE.apply(data)[0]

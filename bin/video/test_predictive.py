import os

import cv2
from matplotlib import pyplot as plt

from src.video.utils import (
    decode_video_delta,
    decode_video_delta_predictive,
    encode_video,
    encode_video_delta,
    encode_video_delta_predictive,
    video_reader,
)


def encode_video_h264(input_file, output_file, codec="h264", fps=30.0, max_frames=20):
    # Open the video reader
    video_cap = cv2.VideoCapture(input_file)

    # Get video dimensions and create VideoWriter
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*codec), fps, (width, height)
    )

    for count, frames in enumerate(video_reader(input_file)):
        if count == max_frames:
            break
        # Encode frames using h264
        encoded_frames = cv2.cvtColor(frames, cv2.COLOR_YCrCb2BGR)
        video_writer.write(encoded_frames)

    # Release the video reader and writer
    video_cap.release()
    video_writer.release()


def main():
    FRAMES = 10
    COMPRESSION_LEVEL = 1
    for video_path in os.listdir("data_video"):
        video_frames = video_reader("data_video/" + video_path, FRAMES)

        encoded_video_delta_predictive = encode_video_delta_predictive(
            video_frames, compression_level=COMPRESSION_LEVEL, reset_every=200
        )

        decoded_video_delta_predictive = decode_video_delta_predictive(
            encoded_video_delta_predictive,
        )

        # Plot the decoded video frame by frame
        for frame in decoded_video_delta_predictive:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_YCrCb2RGB))
            plt.show()
            input()

if __name__ == "__main__":
    main()
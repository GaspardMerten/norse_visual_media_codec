import os
from cmath import log10, sqrt

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.video.utils import (
    decode_video_delta, decode_video_delta_predictive, encode_video,
    encode_video_delta,
    encode_video_delta_predictive,
    video_reader,
)


def calculate_psnr(original, compressed):
    mse = np.mean((np.array(list(original)) - np.array(list(compressed))) ** 2)
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def encode_video_h264(input_file, output_file, codec='h264', fps=30.0, max_frames=20):
    # Open the video reader
    video_cap = cv2.VideoCapture(input_file)

    # Get video dimensions and create VideoWriter
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*codec), fps, (width, height))

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
    FRAMES = 16
    COMPRESSION_LEVEL = 2
    for video_path in os.listdir("data_video"):
        video_frames = video_reader("data_video/" + video_path, FRAMES)
        original_video = [cv2.cvtColor(frame, cv2.COLOR_YUV2RGB) for frame in video_reader("data_video/" + video_path, FRAMES)]

        encoded_video_delta_predictive = encode_video_delta_predictive(
            video_frames, compression_level=COMPRESSION_LEVEL, reset_every=8
        )

        c = decode_video_delta_predictive(encoded_video_delta_predictive)

        c = [cv2.cvtColor(frame, cv2.COLOR_YUV2RGB) for frame in c]

        print("PSNR", calculate_psnr(original_video, c))

        for i in range(FRAMES):
            plt.imshow(c[i])
            plt.show()
            input()


        break

        print("Encoded video with delta predictive")
        video_frames = video_reader("data_video/" + video_path, FRAMES)

        encoded_video_delta = encode_video_delta(
            video_frames, compression_level=COMPRESSION_LEVEL, reset_every=200
        )
        print("Encoded video with delta")

        video_frames = video_reader("data_video/" + video_path, FRAMES)

        encoded_video_h = encode_video(video_frames, compression_level=COMPRESSION_LEVEL)
        print("Encoded video with huffman")

        def convert_yuv_to_rgb(yuv):
            # Convert YCrCb to RGB
            return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)

        a = list(video_reader("data_video/" + video_path, FRAMES))
        print("Decoded original video")
        b = decode_video_delta(encoded_video_delta)
        print("Decoded delta video")
        print("Decoded delta predictive video")


        # Reset plot size
        plt.rcParams["figure.figsize"] = (6, 4)


        with open("output_video/" + video_path + ".encoded", "wb") as f:
            f.write(encoded_video_h)

        with open("output_video/" + video_path + ".encoded_delta", "wb") as f:
            f.write(encoded_video_delta)

        with open("output_video/" + video_path + ".encoded_delta_predictive", "wb") as f:
            f.write(encoded_video_delta_predictive)


        # Get size of each encoded video on disk
        size = os.path.getsize("output_video/" + video_path + ".encoded")
        size_delta = os.path.getsize("output_video/" + video_path + ".encoded_delta")
        size_delta_predictive = os.path.getsize("output_video/" + video_path + ".encoded_delta_predictive")
        size_h264 = os.path.getsize("output_video/" + video_path + ".mp4")
        original_size = os.path.getsize("data_video/" + video_path)

        print(f"Original size: {original_size} bytes")
        print(f"Size: {size} bytes")
        print(f"Size delta: {size_delta} bytes")
        print(f"Size delta predictive: {size_delta_predictive} bytes")
        print(f"Size h264: {size_h264} bytes")

        # Plot difference in size
        plt.bar(
            ["Normal", "Delta", "Delta Predictive", "H264"],
            [size, size_delta, size_delta_predictive, size_h264],
            )
        plt.title("Difference in size")
        plt.show()


if __name__ == "__main__":
    main()

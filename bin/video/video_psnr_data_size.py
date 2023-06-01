import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import log10, sqrt

from src.video.utils import decode_video_delta_predictive, encode_video_delta_predictive, video_reader


def calculate_psnr(original, compressed):
    mse = np.mean((np.array(list(original)) - np.array(list(compressed))) ** 2)
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    for video_path in os.listdir("data_video"):
        psnrs = []
        sizes = []
        compression_levels = []

        FRAMES = 10
        original_video = list(video_reader("data_video/" + video_path, FRAMES))
        original_video = [cv2.cvtColor(frame, cv2.COLOR_YUV2RGB) for frame in original_video]

        for level in [0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print(f"Encoding {video_path} with compression level {level}")
            video_frames = video_reader("data_video/" + video_path, FRAMES)

            encoded_video = encode_video_delta_predictive(video_frames, compression_level=level, reset_every=5)
            print(len(encoded_video))

            with open("output_video/" + video_path + ".encoded", "wb") as f:
                f.write(encoded_video)

            decoded_frames = decode_video_delta_predictive(encoded_video)

            # Convert decoded frames to rgb
            decoded_frames = [cv2.cvtColor(frame, cv2.COLOR_YUV2RGB) for frame in decoded_frames]

            psnr = calculate_psnr(original_video, decoded_frames)
            size = len(encoded_video)

            print(f"PSNR: {psnr} dB", f"Size: {size} bytes")

            psnrs.append(psnr)
            compression_levels.append(level)
            sizes.append(size)  # image size in bytes

        # Plot rate-distortion curve per size and compression level (next to each other)
        plt.figure()
        # Increase dpi to make the plot clearer
        plt.figure(dpi=200)
        # Increase plot size (so legend is not cut off)
        plt.gcf().set_size_inches(10, 5)

        plt.subplot(1, 2, 1)
        plt.plot(compression_levels, psnrs, label="PSNR vs Compression Level")
        plt.xlabel("Compression Level")
        plt.ylabel("PSNR (dB)")
        plt.title(f"Rate-Distortion Curve for {video_path}")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(sizes, psnrs, label="PSNR vs Data Size")
        plt.xlabel("Data Size (bytes)")
        plt.ylabel("PSNR (dB)")
        plt.title(f"Rate-Distortion Curve for {video_path}")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()

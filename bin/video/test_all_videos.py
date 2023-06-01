import os

from matplotlib import pyplot as plt

from src.video.utils import (
    encode_video,
    encode_video_delta,
    encode_video_delta_predictive,
    video_reader,
)


def main():
    FRAMES = 20
    COMPRESSION_LEVEL = 1
    for video_path in os.listdir("data_video"):
        video_frames = video_reader("data_video/" + video_path, FRAMES)

        encoded_video_delta_predictive = encode_video_delta_predictive(
            video_frames, compression_level=COMPRESSION_LEVEL, reset_every=5
        )
        video_frames = video_reader("data_video/" + video_path, FRAMES)

        encoded_video_delta = encode_video_delta(
            video_frames, compression_level=COMPRESSION_LEVEL, reset_every=5
        )

        video_frames = video_reader("data_video/" + video_path, FRAMES)

        encoded_video = encode_video(video_frames, compression_level=COMPRESSION_LEVEL)

        with open("output_video/" + video_path + ".encoded", "wb") as f:
            f.write(encoded_video)

        with open("output_video/" + video_path + ".encoded_delta", "wb") as f:
            f.write(encoded_video_delta)


        # Get size of each encoded video on disk
        size = os.path.getsize("output_video/" + video_path + ".encoded")
        size_delta = os.path.getsize("output_video/" + video_path + ".encoded_delta")
        size_delta_predictive = os.path.getsize("output_video/" + video_path + ".encoded_delta_predictive")
        original_size = os.path.getsize("data_video/" + video_path)

        print(f"Original size: {original_size} bytes")
        print(f"Size: {size} bytes")
        print(f"Size delta: {size_delta} bytes")
        print(f"Size delta predictive: {size_delta_predictive} bytes")

        # Plot difference in size
        plt.bar(
            ["Normal", "Delta", "Delta Predictive"],
            [size, size_delta, size_delta_predictive],
        )
        plt.title("Difference in size")
        plt.show()

if __name__ == "__main__":
    main()

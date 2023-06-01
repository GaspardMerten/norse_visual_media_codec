import os
from math import log10, sqrt

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from src.image import norse_image_decoder, norse_image_encoder


def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # same image case
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


MAX_REACHABLE_PSNR = 20 * log10(255)


def main():
    for image_path in os.listdir("data"):
        psnrs = []
        sizes = []

        base_image = np.array(Image.open("data/" + image_path))
        base_image_size = os.path.getsize("data/" + image_path)
        decoded_images = []
        encoded_image_sizes = []

        compression_levels = [1, 2, 3, 4, 5]

        for level in compression_levels:
            encoded_image = norse_image_encoder(base_image, compression_level=level)

            with open("output/" + image_path + ".norse", "wb") as f:
                f.write(encoded_image)

            # Add size on disk of encoded image
            encoded_image_sizes.append(
                os.path.getsize("output/" + image_path + ".norse")
            )

            decoded_image = norse_image_decoder(encoded_image)
            decoded_images.append(decoded_image)

            psnrs.append(calculate_psnr(base_image, decoded_image))
            sizes.append(len(encoded_image))  # image size in bytes

        # Display O.1 and 1 compression level images and original image
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.imshow(base_image)
        plt.title("Original Image")
        plt.axis("off")
        for i in range(len(compression_levels[:2])):
            plt.subplot(1, 3, i + 2)
            plt.imshow(decoded_images[i])
            plt.title(f"Decoded Image (PSNR: {psnrs[i]:.2f} dB)")
            plt.axis("off")

        # Plot rate-distortion curve
        plt.figure()

        plt.plot(sizes, psnrs, label="PSNR vs Data Size")
        plt.xlabel("Data Size (bytes)")
        plt.ylabel("PSNR (dB)")
        plt.title(f"Rate-Distortion Curve for {image_path}")
        # Add maximum reachable PSNR line
        plt.axhline(
            y=MAX_REACHABLE_PSNR,
            color="r",
            linestyle="--",
            label="Maximum Reachable PSNR",
        )
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot compression level vs PSNR
        plt.figure()
        plt.plot(compression_levels, psnrs, label="PSNR vs Compression Level")
        plt.xlabel("Compression Level")
        plt.ylabel("PSNR (dB)")
        plt.title(f"Compression Level vs PSNR for {image_path}")
        plt.axhline(
            y=MAX_REACHABLE_PSNR,
            color="r",
            linestyle="--",
            label="Maximum Reachable PSNR",
        )

        # Add a dashed vertical line at compression level 1 (Indicating the default compression level)
        plt.axvline(x=1, color="g", linestyle="--", label="Default Compression Level")

        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the ratio between the original image size and the compressed image size
        plt.plot(
            compression_levels,
            [100 * size / base_image_size for size in encoded_image_sizes],
            label="Compression Percentage",
        )
        plt.grid(True)
        plt.ylabel("Compression Percentage")
        plt.title(f"Compression Level vs Compression Percentage for {image_path}")
        # Annotate first compression level with the compression ratio
        plt.annotate(
            f"Compression Ratio: {100 * encoded_image_sizes[0] / base_image_size:.2f}%",
            xy=(1, 100 * encoded_image_sizes[0] / base_image_size),
        )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()

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


def main():
    norse_sizes = []
    jpeg_sizes = []
    colored_images = []
    jpeg_pnsrs = []
    norse_pnsrs = []

    for image_path in os.listdir("data"):
        color_image = np.array(Image.open("data/" + image_path))
        # Save image as jpeg to output folder.
        jpeg_image_path = "output/" + image_path + ".jpeg"
        Image.fromarray(color_image).save(jpeg_image_path, "JPEG")
        jpeg_sizes.append(os.path.getsize(jpeg_image_path))
        encoded_image = norse_image_encoder(color_image, compression_level=1)
        norse_sizes.append(len(encoded_image))
        colored_images.append(len(color_image.shape) == 3)

        jpeg_array = np.array(Image.open(jpeg_image_path))

        jpeg_pnsr = calculate_psnr(color_image, jpeg_array)
        decoded_image = norse_image_decoder(encoded_image)


        # Save decoded image to output folder.
        decoded_image_path = "output/" + image_path + ".norse.jpeg"
        Image.fromarray(decoded_image).save(decoded_image_path, "JPEG")



        norse_pnsr = calculate_psnr(color_image, decoded_image)

        jpeg_pnsrs.append(jpeg_pnsr)
        norse_pnsrs.append(norse_pnsr)

    # For each index, calculate the ratio of the norse size to the jpeg size.
    ratios = [norse_sizes[i] / jpeg_sizes[i] for i in range(len(norse_sizes))]

    # Plot boxplot for jpeg PNSRS and norse PNSRS.
    plt.figure()
    plt.boxplot([jpeg_pnsrs, norse_pnsrs], labels=["JPEG PSNR", "Norse PSNR"])
    plt.title("JPEG PSNR vs Norse PSNR")
    plt.ylabel("PSNR (dB)")
    plt.show()

    # Plot jpeg pnsrs vs norse pnsrs
    plt.figure()
    plt.scatter(
        jpeg_pnsrs,
        norse_pnsrs,
        c=["red" if colored_images[i] else "blue" for i in range(len(colored_images))],
        label="JPEG PSNR vs Norse PSNR",
    )
    # Add x = y line.
    plt.plot([0, 100], [0, 100], color="black", label="x = y")
    plt.xlabel("JPEG PSNR (dB)")
    plt.ylabel("Norse PSNR (dB)")
    plt.title("JPEG PSNR vs Norse PSNR for Each Image")
    plt.legend()

    # Plot the ratios.
    plt.figure()

    # Use red for colored images and blue for    grayscale images.
    plt.scatter(
        [i for i in range(len(ratios))],
        ratios,
        c=["red" if colored_images[i] else "blue" for i in range(len(colored_images))],

    )

    plt.xlabel("Image Index")
    plt.ylabel("Ratio")
    plt.title("Norse Size / JPEG Size for Each Image")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

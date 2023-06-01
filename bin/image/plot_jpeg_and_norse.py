import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from src.image import norse_image_decoder, norse_image_encoder


def plot_jpeg_vs_norse():
    for image_path in os.listdir("data"):
        color_image = np.array(Image.open("data/" + image_path))
        # Save image as jpeg to output folder.
        jpeg_image_path = "output/" + image_path + ".jpeg"
        Image.fromarray(color_image).save(jpeg_image_path, "JPEG")
        encoded_image = norse_image_encoder(color_image, compression_level=1)
        decoded_image = norse_image_decoder(encoded_image)
        jpeg_array = np.array(Image.open(jpeg_image_path))

        # Save decoded image as jpeg to output folder.
        norse_image_path = "output/" + image_path + ".norse.x.tiff"
        Image.fromarray(decoded_image).save(norse_image_path, "TIFF")

        # Save jpeg image as jpeg to output folder.
        jpeg_image_path = "output/" + image_path + ".jpeg"
        Image.fromarray(jpeg_array).save(jpeg_image_path, "JPEG")

        # Encoded image to RGB. (from YCbCr)
        if len(decoded_image.shape) == 3:
            decoded_image = np.array(Image.fromarray(decoded_image).convert("RGB"))
            color_image = np.array(Image.fromarray(color_image).convert("RGB"))
        else:
            # grayscale
            plt.gray()



        # Plot the JPEG image,next to the norse image.
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(jpeg_array)
        plt.title("JPEG")
        plt.subplot(1, 3, 2)
        plt.imshow(decoded_image)
        plt.title("Norse")
        plt.subplot(1, 3, 3)
        plt.imshow(color_image)
        plt.title("Original")
        plt.show()


if __name__ == "__main__":
    plot_jpeg_vs_norse()
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from src.image.pipelines import (
    COLOR_IMAGE_DECODING_PIPELINE,
    COLOR_IMAGE_ENCODING_PIPELINE,
    SINGLE_LAYER_DECODING_PIPELINE_Y_OR_GRAY,
    SINGLE_LAYER_ENCODING_PIPELINE_Y_OR_GRAY,
)


def norse_image_encoder(data: np.ndarray, compression_level=1) -> bytes:
    """
    This function encodes an image using the Norse JPEG encoder.
    :param compression_level: The compression level to be used (the higher the
    compression level, the more compressed the image will be).
    :param data:  The image to be encoded in the form of a numpy array (if color,
    it should be in YCbCr format).
    :return: The encoded image in the form of a byte array.
    """
    assert (compression_level >= 0.1) and (compression_level <= 100)

    color_tag = str(1 if len(data.shape) == 3 else 0).encode("utf-8")

    # If image is grayscale, apply simple pipeline.
    if len(data.shape) == 2:
        output, headers = SINGLE_LAYER_ENCODING_PIPELINE_Y_OR_GRAY.apply(data, dict(compression_level=compression_level))
    else:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2YCR_CB)
        # for each layer, print min and max values.
        for i in range(data.shape[2]):
            print(data[:, :, i].min())
            print(data[:, :, i].max())

        output, headers = COLOR_IMAGE_ENCODING_PIPELINE.apply(data, dict(compression_level=compression_level))

    return color_tag + output


def norse_image_decoder(data: bytes) -> np.ndarray:
    """
    This function decodes an image using the Norse JPEG decoder.
    :param data: The image to be decoded in the form of a byte array.
    :return: The decoded image in the form of a numpy array (if color, it will be
    in YCbCr format).
    """

    # Check if first byte is 1, which indicates color image.
    is_color = data[0] == 49

    # If image is grayscale, apply simple pipeline.
    if is_color:
        output = COLOR_IMAGE_DECODING_PIPELINE.apply(data[1:])[0]

    else:
        output = SINGLE_LAYER_DECODING_PIPELINE_Y_OR_GRAY.apply(data[1:])[0]

    return cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB)


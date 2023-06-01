import gzip
import json
from itertools import product
from typing import List

import cv2
import numpy as np
from scipy.fftpack import idct
from skimage.color import ycbcr2rgb

from src.shared.shared import Element, Pipeline
from src.shared.tables import INV_HEADERS_TABLE


class InverseDeltaLayers(Element):
    def __init__(self):
        super().__init__(self._inverse_delta_layers, "Inverse delta layers")

    @staticmethod
    def _inverse_delta_layers(image: np.ndarray) -> np.ndarray:
        """Calculates the inverse difference between each layer and the previous one.

        Args:
            image (np.ndarray): Image to calculate the difference for.

        Returns:
            np.ndarray: Image with the difference between each layer and the previous one.
        """

        # Add third layer to second layer
        image[:, :, 2] += image[:, :, 1]

        return image


class ConvertToRGB(Element):
    def __init__(self):
        super().__init__(self._convert_to_rgb, "Convert to RGB")

    @staticmethod
    def _convert_to_rgb(image: np.ndarray) -> np.ndarray:
        return ycbcr2rgb(image)


class CombineBlocks(Element):
    def __init__(self, block_size: int):
        self.block_size = block_size
        super().__init__(
            self._combine_blocks, "Combine blocks - size {}".format(block_size)
        )

    @staticmethod
    def _combine_blocks(image: np.ndarray) -> np.ndarray:
        # Reshape image from width/window_size x height/window_size x window_size x window_size
        # to width x height
        width, height, window_size, _ = image.shape
        output = np.zeros((width * window_size, height * window_size), dtype=int)

        for i in range(width):
            for j in range(height):
                output[
                    i * window_size : (i + 1) * window_size,
                    j * window_size : (j + 1) * window_size,
                ] = image[i, j]

        return output


class IDCT(Element):
    def __init__(self):
        super().__init__(self._idct, "IDCT")

    @staticmethod
    def _idct(image: np.ndarray) -> np.ndarray:
        output = np.zeros(image.shape)
        for i, j in product(range(image.shape[0]), range(image.shape[1])):
            output[i, j] = idct(idct(image[i, j].T, norm="ortho").T, norm="ortho")
            # Remove values below 0 and above 255
            output[i, j] = np.clip(output[i, j], 0, 255)

        # Round all values to integers
        return np.round(output).astype(int)


class InverseQuantization(Element):
    def __init__(self, quantization_table):
        super().__init__(self._inverse_quantize, "Inverse Quantization")
        self.quantization_table = quantization_table

    def _inverse_quantize(self, dct_coefficients):
        compression_level = self.headers.get("compression_level", 1)
        return (dct_coefficients * self.quantization_table * compression_level).astype(
            float
        )


class HuffmanDecoding(Element):
    def __init__(self, to_bytes=False):
        self.to_bytes = to_bytes
        super().__init__(self._huffman_decoding, "Huffman decoding")

    def _huffman_decoding(self, data: List[int]) -> List[int]:
        binary_length = self.headers["binary_length"]
        huffman_dict = self.headers["huffman"]
        # Reverse the huffman dictionary
        huffman_dict = {v: k for k, v in huffman_dict.items()}
        # Convert the encoded bytes back to binary string
        bin_string = "".join([bin(b)[2:].zfill(8) for b in data[:-1]])
        missing_length = binary_length - len(bin_string)
        bin_string += bin(data[-1])[2:].zfill(missing_length)
        # Trim the binary string to its original length
        bin_string = bin_string[:binary_length]

        output = []
        buffer = ""

        for char in bin_string:
            buffer += char
            if buffer in huffman_dict:
                output.append(int(huffman_dict[buffer]))
                buffer = ""

        if self.to_bytes:
            output = bytes(output)

        return output


class UnFlatten(Element):
    def __init__(self):
        super().__init__(self._unflatten, "Unflatten")

    def _unflatten(self, data: List[int]) -> List[int]:
        original_shape = self.headers["flatten_shape"]

        # Reshape the decoded array into the original image shape
        decoded = np.array(data).reshape(original_shape)

        return decoded


class RunLengthDecoding(Element):
    def __init__(self):
        super().__init__(self._run_length_decoding, "Run length decoding")

    def _run_length_decoding(self, data: List[int]) -> List[int]:
        decoded = []

        previous_was_zero = False

        # For each number in the input array, if it is a zero, then the next number is the run length
        for i in range(0, len(data) - 1):
            if data[i] == 0:
                decoded.extend([0] * data[i + 1])
                previous_was_zero = True
            else:
                if previous_was_zero:
                    previous_was_zero = False
                else:
                    decoded.append(data[i])
        # Add last element if it is not a zero
        if data[-2] != 0:
            decoded.append(data[-1])

        return decoded

    @staticmethod
    def _run_length_decoding_for_block(encoded_block: List[int]) -> List[int]:
        output = []

        i = 0
        while i < len(encoded_block):
            if (
                encoded_block[i] == 0
            ):  # check if the number is a zero indicating a run of zeros
                count = encoded_block[i + 1]
                output.extend([0] * count)  # add 'count' zeros to the output
                i += 2  # skip the next number which indicates the run length
            else:
                output.append(encoded_block[i])
                i += 1  # increment index

        return output


class ZigZagDecoding(Element):
    def __init__(self):
        super().__init__(self._zigzag_decoding, "ZigZag decoding")

    def _zigzag_decoding(self, encoded_image: np.ndarray) -> np.ndarray:
        # Assume the shape of encoded_image is (width/block_size, height/block_size, block_size*block_size)
        shape = encoded_image.shape
        window_size = int(np.sqrt(shape[2]))
        output = np.zeros((shape[0], shape[1], window_size, window_size))
        for i, j in product(range(shape[0]), range(shape[1])):
            output[i, j] = self._zigzag_decoding_for_block(
                encoded_image[i, j], window_size
            )

        return output

    @staticmethod
    def _zigzag_decoding_for_block(
        encoded_block: np.ndarray, window_size: int
    ) -> np.ndarray:
        output = np.zeros((window_size, window_size))

        row, col = 0, 0
        direction = 1

        for i in range(window_size * window_size):
            output[row, col] = encoded_block[i]

            if direction == 1:  # Moving diagonally up
                if col == window_size - 1:
                    row += 1
                    direction = -1
                elif row == 0:
                    col += 1
                    direction = -1
                else:
                    row -= 1
                    col += 1
            else:  # Moving diagonally down
                if row == window_size - 1:
                    col += 1
                    direction = 1
                elif col == 0:
                    row += 1
                    direction = 1
                else:
                    row += 1
                    col -= 1

        return output


class GZIPDecoder(Element):
    def __init__(self):
        super().__init__(self._bytes_to_gzip_string, "UNGZIP Bytes")

    @staticmethod
    def _bytes_to_gzip_string(data: bytes) -> bytes:
        return gzip.decompress(data)


class IntListDecoder(Element):
    def __init__(self, with_headers: bool = True):
        self.with_headers = with_headers
        super().__init__(self._bytes_to_int_list, "Bytes to int list")

    def _bytes_to_int_list(self, data: bytes) -> List[int]:
        # Split the data at \n (bytes) and load first element as the header (json.loads) and the rest as the data
        data = data.split(b"\nVIC\n")

        headers = json.loads(data[0].decode("utf-8"))
        if self.with_headers:
            # Replace all keys using INVERSE_HEADERS_TABLE
            headers = {INV_HEADERS_TABLE[key]: value for key, value in headers.items()}

            self.headers.update(headers)

        return list(data[1])


class SubsampleDecoder(Element):
    def __init__(self, block_size: int):
        self.block_size = block_size
        super().__init__(self._oversample, "Oversample")

    def _oversample(self, subsampled_image: np.ndarray) -> np.ndarray:
        rounded_shape = self.headers["pre_subsample_shape"]
        zoom_factor = (self.block_size, self.block_size)
        oversampled_image = cv2.resize(
            subsampled_image,
            (rounded_shape[1], rounded_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        return oversampled_image


class DeStandardizeShape(Element):
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        super().__init__(self._destandardize_shape, "De-standardize shape")

    def _destandardize_shape(self, image: np.ndarray) -> np.ndarray:
        original_shape = self.headers.get(f"{self.prefix}original_shape")
        if original_shape is None:
            raise ValueError("Missing 'original_shape' in headers")

        # Remove extra rows
        if original_shape[0] < image.shape[0]:
            image = image[: original_shape[0], :]

        # Remove extra columns
        if original_shape[1] < image.shape[1]:
            image = image[:, : original_shape[1]]

        return image.astype(np.uint8)


class ReconstructLayersDecoder(Element):
    def __init__(self, pipelines: List[Pipeline]):
        self.pipelines = pipelines
        super().__init__(self._reconstruct_layers, "Reconstruct Layers")

    def _reconstruct_layers(self, layers: List[bytes]) -> np.ndarray:
        """Reconstructs the distributed layers into a single image.

        Args:
            distributed_image (bytes): Distributed layers.

        Returns:
            np.ndarray: Reconstructed image.
        """

        # Split the distributed layers using double newline separator
        output = []

        for index, pipeline in enumerate(self.pipelines):
            result, headers = pipeline.apply(
                layers[index], headers=self.headers, run_settings=self.run_settings
            )
            self.headers.update(headers)
            output.append(result)

        shape = output[0].shape

        # Create a new array with the shape of the original image
        reconstructed_image = np.zeros(
            (shape[0], shape[1], len(output)), dtype=np.uint8
        )

        # Add each layer to the new array
        for i, layer in enumerate(output):
            reconstructed_image[:, :, i] = layer

        return reconstructed_image

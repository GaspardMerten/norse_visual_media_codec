import gzip
import json
from collections import defaultdict
from itertools import chain, product
from typing import Any, Callable, Dict, List

import numpy as np
from scipy.fftpack import dct
from skimage.color import rgb2ycbcr
from skimage.util import view_as_blocks

from src.image.huffman import get_huffman_dict_for_frequency
from src.shared.shared import Element, Pipeline
from src.shared.tables import HEADERS_TABLE


class DeltaLayers(Element):
    def __init__(self):
        super().__init__(self._delta_layers, "Delta layers")

    @staticmethod
    def _delta_layers(image: np.ndarray) -> np.ndarray:
        """Calculates the difference between each layer and the previous one.

        Args:
            image (np.ndarray): Image to calculate the difference for.

        Returns:
            np.ndarray: Image with the difference between each layer and the previous one.
        """

        # Diff second layer with third layer
        image[:, :, 2] -= image[:, :, 1]

        return image


class DistributeLayerToPipelines(Element):
    def __init__(self, pipelines: List[Pipeline]):
        self.pipelines = pipelines
        super().__init__(
            self._distribute_layer_to_pipelines, "Distribute layer to pipelines"
        )

    def _distribute_layer_to_pipelines(self, image: np.ndarray) -> List[bytes]:
        """Distributes a layer to multiple pipelines.

        Args:
            image (np.ndarray): Image to distribute.

        Returns:
            np.ndarray: List of images (encoded), each one corresponding to a pipeline.
        """
        output = []
        # For layer Cr, Cb, Y apply corresponding pipeline
        for i, pipeline in enumerate(self.pipelines):
            result, headers = pipeline.apply(image[:, :, i], self.run_settings)
            self.headers.update(headers)
            output.append(result)

        return output


class LinesDelta(Element):
    def __init__(self):
        super().__init__(self._lines_delta, "Lines delta")

    @staticmethod
    def _lines_delta(image: np.ndarray) -> np.ndarray:
        """Calculates the difference between each line and the previous one.

        Args:
            image (np.ndarray): Image to calculate the difference for.

        Returns:
            np.ndarray: Image with the difference between each line and the previous one.
        """

        # Np diff but keep the first line
        return np.concatenate(
            (image[:1], np.diff(image, axis=0)),
            axis=0,
        )


class ConvertToYCbCr(Element):
    def __init__(self):
        super().__init__(self._convert_to_ycbcr, "Convert to YCbCr")

    @staticmethod
    def _convert_to_ycbcr(image: np.ndarray) -> np.ndarray:
        return rgb2ycbcr(image)


class ApplyForEach(Element):
    def __init__(self, element: Callable, name: str, dimension: int):
        self.element = element
        self.name = name
        self.dimension = dimension

        super().__init__(self.apply, "Apply for each")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Applies a function for each element in an image.

        Args:
            image (np.ndarray): Image to apply the function to.

        Returns:
            np.ndarray: Image with the function applied.
        """

        return np.apply_along_axis(self.function, self.dimension, image)

    def __str__(self):
        return self.name


class SplitToBlocks(Element):
    def __init__(self, block_size: int):
        self.block_size = block_size
        super().__init__(
            self._split_to_blocks, "Split to blocks - size {}".format(block_size)
        )

    def _split_to_blocks(self, image: np.ndarray) -> np.ndarray:
        """Splits an image into blocks.

        Args:
            image (np.ndarray): Image to split.

        Returns:
            np.ndarray: Image blocks.
        """
        return view_as_blocks(image, block_shape=(self.block_size, self.block_size))


class DCT(Element):
    def __init__(self):
        super().__init__(self._dct, "DCT")

    @staticmethod
    def _dct(image: np.ndarray) -> np.ndarray:
        # Expected input shape: (width/block_size, height/block_size, block_size, block_size)
        output = np.zeros(image.shape)
        for i, j in product(range(image.shape[0]), range(image.shape[1])):
            source = image[i, j].T.astype(np.uint8)
            first_level = dct(source, norm="ortho").T
            output[i, j] = dct(first_level, norm="ortho")

        return output


class Quantization(Element):
    def __init__(self, quantization_table):
        super().__init__(self._quantize, "Quantization")
        self.quantization_table = quantization_table

    def _quantize(self, dct_coefficients):
        """
        Quantizes DCT coefficients.

        Args:
            dct_coefficients (np.ndarray): DCT coefficients to quantize.

        Returns:
            np.ndarray: Quantized DCT coefficients.
        """

        compression_level = self.run_settings.get("compression_level", 1)
        self.headers["compression_level"] = compression_level

        return np.round(
            dct_coefficients / (self.quantization_table * compression_level)
        ).astype(int)


class ZigZag(Element):
    def __init__(self):
        super().__init__(self._zigzag, "ZigZag")

    def _zigzag(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape
        # Expected input shape: (width/block_size, height/block_size, block_size,block_size)
        output = np.zeros(shape[:2] + (shape[2] * shape[3],))

        # Apply _zigzag_for_block to each block (use last two dimensions)
        for i, j in product(range(shape[0]), range(shape[1])):
            output[i, j] = self._zigzag_for_block(image[i, j])
        return output

    @staticmethod
    def _zigzag_for_block(block: np.ndarray) -> list:
        """Applies the zigzag algorithm to a block.

        Args:
            block (np.ndarray): Block to apply the algorithm to.

        Returns:
            np.ndarray: Block with the zigzag algorithm applied.
        """
        # Expected input shape: (block_size, block_size)
        size = block.shape[0]
        solution = [[] for _ in range(2 * size - 1)]

        for i in range(size):
            for j in range(size):
                summed_values = i + j
                solution[summed_values].append(block[i][j])

        for i in range(0, len(solution), 2):
            solution[i] = solution[i][::-1]

        output = list(chain(*solution))
        return output


class Flatten(Element):
    def __init__(self):
        super().__init__(self._flatten, "Flatten")

    def _flatten(self, image: np.ndarray) -> np.ndarray:
        """Flattens an image.

        Args:
            image (np.ndarray): Image to flatten.

        Returns:
            np.ndarray: Flattened image.
        """

        # Store original shape in headers
        original_shape = image.shape
        self.headers["flatten_shape"] = original_shape

        return image.flatten()


class RunLengthEncoding(Element):
    def __init__(self):
        super().__init__(self._run_length_encoding, "Run length encoding")

    def _run_length_encoding(self, image: List[int]) -> List[int]:
        output = []

        count = 0

        for element in image:
            if element == 0:
                count += 1
            else:
                if count != 0:
                    output.extend([0, count])
                    count = 0

                output.append(int(element))

        if count != 0:
            output.extend([0, count])
        return output


class HuffmanEncoding(Element):
    def __init__(self):
        super().__init__(self._huffman_encoding, "Huffman encoding")

    def _huffman_encoding(self, image: List[int | bytes]) -> List[int | bytes]:
        frequency: Dict[Any] = defaultdict(int)

        for element in image:
            frequency[element] += 1
        huffman_codes = get_huffman_dict_for_frequency(list(frequency.items()))
        self.headers["huffman"] = huffman_codes

        output = ""

        for element in image:
            value = huffman_codes[element]
            output += value

        # Split string into 8-digit chunks
        byte_strings = [output[i : i + 8] for i in range(0, len(output), 8)]

        # Convert byte strings to integers
        bytes_int = [int(byte_str, 2) for byte_str in byte_strings]

        self.headers["binary_length"] = len(output)

        return bytes_int


class GZIPEncoder(Element):
    def __init__(self):
        super().__init__(self._gzip_string_to_bytes, "GZIP string to bytes")

    @staticmethod
    def _gzip_string_to_bytes(data: bytes) -> bytes:
        return gzip.compress(data)


class IntListEncoder(Element):
    def __init__(self, with_headers: bool = True):
        self.with_headers = with_headers
        super().__init__(self._int_list_to_bytes, "Int list to bytes")

    def _int_list_to_bytes(self, data: List[int]) -> bytes:
        extra = bytes()
        if self.with_headers:
            # Replace all keys using HEADERS_TABLE
            self.headers = {
                HEADERS_TABLE[key]: value for key, value in self.headers.items()
            }
            extra = (json.dumps(self.headers) + "\nVIC\n").encode("utf-8")

        return extra + bytes(data)


class StandardizeShape(Element):
    def __init__(self, block_size: int, prefix: str = ""):
        self.block_size = block_size
        self.prefix = prefix
        super().__init__(self._standardize_shape, "Standardize shape")

    def _standardize_shape(self, image: np.ndarray) -> np.ndarray:
        self.headers[f"{self.prefix}original_shape"] = image.shape[:2]
        self.headers["block_size"] = self.block_size

        missing_rows = (
            self.block_size - image.shape[0] % self.block_size
        ) % self.block_size
        missing_columns = (
            self.block_size - image.shape[1] % self.block_size
        ) % self.block_size

        for _ in range(missing_rows):
            # Insert row of zeros at the end
            image = np.insert(image, image.shape[0], 0, axis=0)

        for _ in range(missing_columns):
            # Insert column of zeros at the end
            image = np.insert(image, image.shape[1], 0, axis=1)

        self.headers["rounded_shape"] = image.shape[:2]

        return image


class SubsampleEncoder(Element):
    def __init__(self, block_size: int):
        self.block_size = block_size
        super().__init__(self._subsample, "Subsample")

    def _subsample(self, image: np.ndarray) -> np.ndarray:
        self.headers["pre_subsample_shape"] = image.shape[:2]

        # Compute the dimensions of the subsampled image
        new_height = image.shape[0] // self.block_size
        new_width = image.shape[1] // self.block_size

        # Initialize an empty array for the subsampled image
        subsampled_image = np.zeros((new_height, new_width), dtype=np.uint8)

        # Compute the mean of each block and assign it to the corresponding pixel in the subsampled image
        for i in range(new_height):
            for j in range(new_width):
                block = image[
                    i * self.block_size : (i + 1) * self.block_size,
                    j * self.block_size : (j + 1) * self.block_size,
                ]
                # Subsample by taking the median of each block
                subsampled_image[i, j] = np.median(block)

        return subsampled_image


class DebugEncoder(Element):
    def __init__(self):
        super().__init__(self._debug, "Debug")

    def _debug(self, image: np.ndarray) -> np.ndarray:
        return image

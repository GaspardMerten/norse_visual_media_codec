import numpy as np

JPEG_Y_QUANTIZATION_TABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

JPEG_C_QUANTIZATION_TABLE = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


HEADERS_TABLE = {
    "original_shape": 0,
    "block_size": 1,
    "rounded_shape": 2,
    "binary_length": 3,
    "huffman": 4,
    "pre_subsample_shape": 5,
    "a_original_shape": 6,
    "compression_level": 7,
    "bits_per_pixel": 8,
    "flatten_shape": 9,
    "distribute_delta_reset_every": 10,
    "predictive_delta_reset_every": 11,
    "predictive_best_match": 12,
}

INV_HEADERS_TABLE = {str(v): k for k, v in HEADERS_TABLE.items()}



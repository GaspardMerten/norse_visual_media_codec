from src.image import decoders, encoders
from src.image.decoders import InverseDeltaLayers, ReconstructLayersDecoder
from src.image.encoders import DeltaLayers, DistributeLayerToPipelines
from src.shared.encoders import BytesArrayDecoder, BytesArrayEncoder
from src.shared.shared import Pipeline
from src.shared.tables import JPEG_C_QUANTIZATION_TABLE, JPEG_Y_QUANTIZATION_TABLE

BLOCK_SIZE = 8

SINGLE_LAYER_ENCODING_PIPELINE_PARTIAL = lambda table: [
    encoders.StandardizeShape(8),
    encoders.SplitToBlocks(8),
    encoders.DCT(),
    encoders.Quantization(table),
]

SINGLE_LAYER_ENCODING_PIPELINE_LOSSLESS = [
    encoders.ZigZag(),
    encoders.Flatten(),
    encoders.RunLengthEncoding(),
    encoders.HuffmanEncoding(),
    encoders.IntListEncoder(),
    encoders.GZIPEncoder(),
]

SINGLE_LAYER_ENCODING_PIPELINE_CHROMA_LOSSY = SINGLE_LAYER_ENCODING_PIPELINE_PARTIAL(
    JPEG_C_QUANTIZATION_TABLE
)
SINGLE_LAYER_ENCODING_PIPELINE_CHROMA = Pipeline(
    [
        *SINGLE_LAYER_ENCODING_PIPELINE_CHROMA_LOSSY,
        *SINGLE_LAYER_ENCODING_PIPELINE_LOSSLESS,
    ]
)

SINGLE_LAYER_ENCODING_PIPELINE_Y_OR_GRAY_LOSSY = SINGLE_LAYER_ENCODING_PIPELINE_PARTIAL(
    JPEG_Y_QUANTIZATION_TABLE
)

SINGLE_LAYER_ENCODING_PIPELINE_Y_OR_GRAY = Pipeline(
    [
        *SINGLE_LAYER_ENCODING_PIPELINE_Y_OR_GRAY_LOSSY,
        *SINGLE_LAYER_ENCODING_PIPELINE_LOSSLESS,
    ]
)

SINGLE_LAYER_DECODING_PIPELINE_LOSSLESS = [
    decoders.GZIPDecoder(),
    decoders.IntListDecoder(),
    decoders.HuffmanDecoding(),
    decoders.RunLengthDecoding(),
    decoders.UnFlatten(),
    decoders.ZigZagDecoding(),
]

SINGLE_LAYER_DECODING_PIPELINE_PARTIAL = lambda table: [
    decoders.InverseQuantization(table),
    decoders.IDCT(),
    decoders.CombineBlocks(BLOCK_SIZE),
    decoders.DeStandardizeShape(),
]

SINGLE_LAYER_DECODING_PIPELINE_CHROMA_LOSSY = SINGLE_LAYER_DECODING_PIPELINE_PARTIAL(
    JPEG_C_QUANTIZATION_TABLE
)

SINGLE_LAYER_DECODING_PIPELINE_Y_OR_GRAY_LOSSY = SINGLE_LAYER_DECODING_PIPELINE_PARTIAL(
    JPEG_Y_QUANTIZATION_TABLE
)

SINGLE_LAYER_DECODING_PIPELINE_CHROMA = Pipeline(
    [
        *SINGLE_LAYER_DECODING_PIPELINE_LOSSLESS,
        *SINGLE_LAYER_DECODING_PIPELINE_CHROMA_LOSSY,
    ]
)

SINGLE_LAYER_DECODING_PIPELINE_Y_OR_GRAY = Pipeline(
    [
        *SINGLE_LAYER_DECODING_PIPELINE_LOSSLESS,
        *SINGLE_LAYER_DECODING_PIPELINE_Y_OR_GRAY_LOSSY,
    ]
)

SUB_SAMPLING_LEVEL = 2

SINGLE_LAYER_ENCODING_PIPELINE_CHROMA_SUBSAMPLING = Pipeline(
    [
        encoders.StandardizeShape(8, prefix="a_"),
        encoders.SubsampleEncoder(block_size=SUB_SAMPLING_LEVEL),
        *SINGLE_LAYER_ENCODING_PIPELINE_CHROMA.elements,
    ]
)

SINGLE_LAYER_DECODING_PIPELINE_CHROMA_SUBSAMPLING = Pipeline(
    [
        *SINGLE_LAYER_DECODING_PIPELINE_CHROMA.elements,
        decoders.SubsampleDecoder(block_size=SUB_SAMPLING_LEVEL),
        decoders.DeStandardizeShape(prefix="a_"),
    ]
)

COLOR_IMAGE_ENCODING_PIPELINE = Pipeline(
    [
        DeltaLayers(),
        DistributeLayerToPipelines(
            pipelines=[
                SINGLE_LAYER_ENCODING_PIPELINE_Y_OR_GRAY,
                SINGLE_LAYER_ENCODING_PIPELINE_CHROMA_SUBSAMPLING,
                SINGLE_LAYER_ENCODING_PIPELINE_CHROMA_SUBSAMPLING,
            ],
        ),
        BytesArrayEncoder(),
        encoders.GZIPEncoder(),
    ]
)

COLOR_IMAGE_DECODING_PIPELINE = Pipeline(
    [
        decoders.GZIPDecoder(),
        BytesArrayDecoder(),
        ReconstructLayersDecoder(
            pipelines=[
                SINGLE_LAYER_DECODING_PIPELINE_Y_OR_GRAY,
                SINGLE_LAYER_DECODING_PIPELINE_CHROMA_SUBSAMPLING,
                SINGLE_LAYER_DECODING_PIPELINE_CHROMA_SUBSAMPLING,
            ],
        ),
        InverseDeltaLayers(),
    ]
)

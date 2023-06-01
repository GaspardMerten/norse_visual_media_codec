from src.image import (
    COLOR_IMAGE_DECODING_PIPELINE,
    COLOR_IMAGE_ENCODING_PIPELINE,
)
from src.image.decoders import ReconstructLayersDecoder
from src.image.encoders import DistributeLayerToPipelines
from src.image.pipelines import (
    SINGLE_LAYER_DECODING_PIPELINE_CHROMA_LOSSY,
    SINGLE_LAYER_DECODING_PIPELINE_LOSSLESS,
    SINGLE_LAYER_DECODING_PIPELINE_Y_OR_GRAY_LOSSY,
    SINGLE_LAYER_ENCODING_PIPELINE_CHROMA_LOSSY,
    SINGLE_LAYER_ENCODING_PIPELINE_LOSSLESS,
    SINGLE_LAYER_ENCODING_PIPELINE_Y_OR_GRAY_LOSSY,
)
from src.shared.encoders import BytesArrayDecoder, BytesArrayEncoder
from src.shared.shared import Pipeline
from src.video.decoders import DistributedDeltaDecoder, PredictiveDistributedDeltaDecoder
from src.video.encoders import (
    ApplyOnIterable,
    ApplyOnIterableWithMemory,
    DistributedDeltaEncoder,
    PredictiveDistributedDeltaEncoder,
)

LINEAR_VIDEO_ENCODING_PIPELINE = Pipeline(
    [
        ApplyOnIterable(
            pipeline=COLOR_IMAGE_ENCODING_PIPELINE,
        ),
        BytesArrayEncoder(),
    ]
)

LINEAR_VIDEO_DECODING_PIPELINE = Pipeline(
    [
        BytesArrayDecoder(),
        ApplyOnIterable(
            pipeline=COLOR_IMAGE_DECODING_PIPELINE,
        ),
    ]
)

DELTA_VIDEO_ENCODING_LOSSY = ApplyOnIterable(
    pipeline=Pipeline(
        [
            DistributeLayerToPipelines(
                pipelines=[
                    Pipeline(SINGLE_LAYER_ENCODING_PIPELINE_Y_OR_GRAY_LOSSY),
                    Pipeline(SINGLE_LAYER_ENCODING_PIPELINE_CHROMA_LOSSY),
                    Pipeline(SINGLE_LAYER_ENCODING_PIPELINE_CHROMA_LOSSY),
                ],
            ),
        ]
    )
)
DELTA_VIDEO_ENCODING_LOSSLESS = ApplyOnIterable(
    pipeline=Pipeline(
        [
            ApplyOnIterable(pipeline=Pipeline(SINGLE_LAYER_ENCODING_PIPELINE_LOSSLESS)),
            BytesArrayEncoder(),
        ]
    )
)

DELTA_VIDEO_ENCODING_PIPELINE = Pipeline(
    [
        DELTA_VIDEO_ENCODING_LOSSY,
        ApplyOnIterableWithMemory(
            headers_prefix="distribute_delta_",
            pipeline=Pipeline([DistributedDeltaEncoder()]),
        ),
        DELTA_VIDEO_ENCODING_LOSSLESS,
        BytesArrayEncoder(),
    ]
)

DELTA_VIDEO_DECODING_LOSSLESS = ApplyOnIterable(
    pipeline=Pipeline(
        [
            BytesArrayDecoder(),
            ApplyOnIterable(
                pipeline=Pipeline(SINGLE_LAYER_DECODING_PIPELINE_LOSSLESS),
            ),
        ],
    )
)
DELTA_VIDEO_DECODING_LOSSY = ApplyOnIterable(
    pipeline=Pipeline(
        [
            ReconstructLayersDecoder(
                pipelines=[
                    Pipeline(SINGLE_LAYER_DECODING_PIPELINE_Y_OR_GRAY_LOSSY),
                    Pipeline(SINGLE_LAYER_DECODING_PIPELINE_CHROMA_LOSSY),
                    Pipeline(SINGLE_LAYER_DECODING_PIPELINE_CHROMA_LOSSY),
                ],
            ),
        ],
    )
)
DELTA_VIDEO_DECODING_PIPELINE = Pipeline(
    [
        BytesArrayDecoder(),
        DELTA_VIDEO_DECODING_LOSSLESS,
        ApplyOnIterableWithMemory(
            headers_prefix="distribute_delta_",
            pipeline=Pipeline([DistributedDeltaDecoder()]),
        ),
        DELTA_VIDEO_DECODING_LOSSY,
    ]
)

PREDICTIVE_DELTA_VIDEO_ENCODING_PIPELINE = Pipeline(
    [
        DELTA_VIDEO_ENCODING_LOSSY,
        ApplyOnIterableWithMemory(
            headers_prefix="predictive_delta_",
            pipeline=Pipeline([PredictiveDistributedDeltaEncoder()]),
        ),
        DELTA_VIDEO_ENCODING_LOSSLESS,
        BytesArrayEncoder(),
    ]
)
PREDICTIVE_DELTA_VIDEO_DECODING_PIPELINE = Pipeline(
    [
        BytesArrayDecoder(),
        DELTA_VIDEO_DECODING_LOSSLESS,
        ApplyOnIterableWithMemory(
            headers_prefix="predictive_delta_",
            pipeline=Pipeline([PredictiveDistributedDeltaDecoder()]),
        ),
        DELTA_VIDEO_DECODING_LOSSY,
    ]
)

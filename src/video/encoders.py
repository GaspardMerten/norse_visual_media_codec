from itertools import product
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error

from src.shared.shared import Element, Pipeline


class ApplyOnIterable(Element):
    def __init__(self, pipeline: Pipeline, name="ApplyOnIterable"):
        super().__init__(self.apply, name)
        self.pipeline = pipeline

    def apply(self, iterable):
        output = []
        for item in iterable:
            result, headers = self.pipeline.apply(
                item, run_settings=self.run_settings, headers=self.headers
            )
            self.headers.update(headers)
            output.append(result)

        return output


DEFAULT_RESET_EVERY = 1_000


class ApplyOnIterableWithMemory(Element):
    def __init__(
        self,
        pipeline,
        name="ApplyOnIterableWithMemory",
        headers_prefix="",
    ):
        super().__init__(self.apply, name)
        self.pipeline = pipeline
        self.headers_prefix = headers_prefix

    def apply(self, iterable):
        reset_every = self.headers.get(
            f"{self.headers_prefix}reset_every",
            self.run_settings.get("reset_every", DEFAULT_RESET_EVERY),
        )
        self.headers[f"{self.headers_prefix}reset_every"] = reset_every
        output = []

        last_input = None
        last_output = None

        for index, item in enumerate(iterable):
            result, headers = self.pipeline.apply(
                item,
                run_settings={
                    **self.run_settings,
                    "input": last_input,
                    "output": last_output,
                },
                headers=self.headers,
            )
            self.headers.update(headers)
            output.append(result)

            if index % reset_every == 0:
                last_input = None
                last_output = None
            else:
                last_input = item
                last_output = result

        return output


class DistributedDeltaEncoder(Element):
    def __init__(self, name="DeltaEncoder"):
        super().__init__(self.apply, name)

    def apply(self, channels: List[np.ndarray]):
        previous_input = self.run_settings.get("input", None)
        if previous_input is None:
            return channels

        output = []

        for index, channel in enumerate(channels):
            delta_frame = channel - previous_input[index]
            output.append(delta_frame)

        return output


class PredictiveDistributedDeltaEncoder(Element):
    def __init__(self, name="PredictiveDistributedDeltaEncoder"):
        super().__init__(self.apply, name)

    def apply(self, channels: List[np.ndarray]):
        previous_input = self.run_settings.get("input", None)
        previous_output = self.run_settings.get("output", None)

        if previous_input is None:
            return channels
        previous_best_match = None

        if previous_output is not None and len(previous_output) > 3:
            previous_best_match = previous_output[3]

        max_abs_delta_x = 16
        max_abs_delta_y = 16

        first_channel = channels[0]
        # Recombine window of first channel into 16x16 blocks
        new_window_size = 16

        ratio = int(new_window_size / first_channel.shape[2])

        reshape_current_first_channel = self.reshape_window(
            first_channel, new_window_size=new_window_size
        )
        reshape_previous_first_channel = self.reshape_window(
            previous_input[0], new_window_size=new_window_size
        )

        # For each block of current and previous first channel, find the best match
        # in the previous first channel
        best_match = np.zeros(
            (*reshape_current_first_channel.shape[:2], 2, 2), dtype=int
        )
        for x, y in product(
            range(reshape_current_first_channel.shape[0]),
            range(reshape_current_first_channel.shape[1]),
        ):
            current_block = reshape_current_first_channel[x, y]
            best_delta_x, best_delta_y = self.find_best_match(
                current_block,
                reshape_previous_first_channel,
                x,
                y,
                max_abs_delta_x,
                max_abs_delta_y,
            )

            best_match[x, y] = [[best_delta_x], [best_delta_y]]

        output = []

        # Now apply on original image
        for index, channel in enumerate(channels):
            new_channel = np.zeros(channel.shape, dtype=channel.dtype)

            for x, y in product(
                range(channel.shape[0]),
                range(channel.shape[1]),
            ):
                new_channel[x, y] = (
                    channel[x, y]
                    - previous_input[index][
                        x + best_match[x // ratio, y // ratio, 0, 0],
                        y + best_match[x // ratio, y // ratio, 1, 0],
                    ]
                )
            output.append(new_channel)
        # For each unique value in best match, print count
        print(np.unique(best_match, return_counts=True))


        output.append(best_match)

        return output

    def find_best_match(
        self,
        current_block,
        previous_first_channel,
        x,
        y,
        max_abs_delta_x,
        max_abs_delta_y,
    ):
        best_delta_x = None
        best_delta_y = None
        best_match_error = None
        for proposed_x, proposed_y in product(
            range(
                max(0, x - max_abs_delta_x),
                min(previous_first_channel.shape[0], x + max_abs_delta_x),
            ),
            range(
                max(0, y - max_abs_delta_y),
                min(previous_first_channel.shape[1], y + max_abs_delta_y),
            ),
        ):
            previous_block = previous_first_channel[proposed_x, proposed_y]

            error = np.mean(np.abs(current_block - previous_block))

            if best_match_error is None or error < best_match_error:
                best_match_error = error
                best_delta_x = proposed_x - x
                best_delta_y = proposed_y - y
        return int(best_delta_x), int(best_delta_y)

    def reshape_window(self, first_channel, new_window_size=16):
        width, height, window_size, _ = first_channel.shape

        window_size_ratio = new_window_size / window_size

        reshaped_image = np.reshape(first_channel, (width, height, -1))
        # Reshape the image to create 16x16 blocks
        halved_width = int(width / window_size_ratio)
        halved_height = int(height / window_size_ratio)
        return np.reshape(
            reshaped_image,
            (
                halved_width,
                halved_height,
                int(window_size * window_size_ratio),
                int(window_size * window_size_ratio),
            ),
        )

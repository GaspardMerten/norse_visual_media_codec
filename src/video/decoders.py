from itertools import product
from typing import List

import numpy as np

from src.shared.shared import Element


class DistributedDeltaDecoder(Element):
    def __init__(self, name="DeltaDecoder"):
        super().__init__(self.apply, name)

    def apply(self, channels: List[np.ndarray]):
        previous_output = self.run_settings.get("output", None)

        if previous_output is None:
            return channels

        output = []

        for index, channel in enumerate(channels):
            output.append(channel + previous_output[index])

        return output


class PredictiveDistributedDeltaDecoder(Element):
    def __init__(self, name="PredictiveDistributedDeltaDecoder"):
        super().__init__(self.apply, name)

    def apply(self, channels: List[np.ndarray]):
        previous_input = self.run_settings.get("output", None)

        if previous_input is None:
            return channels

        best_match = channels.pop(3).astype(int)

        if len(previous_input) == 4:
            best_match = best_match + previous_input[3]

        output = []

        ratio = int(channels[0].shape[0] / best_match.shape[0])

        for index, channel in enumerate(channels):
            new_channel = np.zeros(channel.shape, dtype=channel.dtype)
            for x, y in product(range(channel.shape[0]), range(channel.shape[1])):
                delta_x, delta_y = best_match[x // ratio, y // ratio]
                new_channel[x, y] = (
                        previous_input[index][
                            min(x + delta_x[0], channel.shape[0] - 1), min(y + delta_y[0], channel.shape[1] - 1)]
                        + channel[x, y]
                )

            output.append(new_channel)

        return output

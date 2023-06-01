import struct
from typing import List

from src.shared.shared import Element

__all__ = ["BytesArrayEncoder", "BytesArrayDecoder"]

class BytesArrayEncoder(Element):
    def __init__(self, name="BytesArrayEncoder"):
        super().__init__(self.apply, name)

    @staticmethod
    def apply(data: List[bytes]) -> bytes:
        # Encode length of each byte array (as a 64-bit integer), then add 0x00 as separator
        output = bytes()

        for item in data:
            # struct.pack formats the length as a 64-bit integer ('Q' format character)
            output += struct.pack("Q", len(item)) + item

        return output


class BytesArrayDecoder(Element):
    def __init__(self, name="BytesArrayDecoder"):
        super().__init__(self.apply, name)

    @staticmethod
    def apply(data: bytes) -> List[bytes]:
        output = []
        idx = 0

        while idx < len(data):
            # Get the length of the byte array
            length = struct.unpack("Q", data[idx : idx + 8])[0]
            idx += 8  # Skip the length bytes

            # Read the byte array and append it to the output
            output.append(data[idx : idx + length])

            # Move the index pointer to the start of the next length byte
            idx += length
        return output

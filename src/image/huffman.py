from typing import Any, List, Tuple


class Node:
    def __init__(self, value, frequency):
        self.value = value
        self.frequency = frequency
        self.left = None
        self.right = None


def get_code_for_node(node, code, huffman_dict):
    if node is None:
        return

    if node.value is not None:
        huffman_dict[node.value] = code
    else:
        get_code_for_node(node.left, code + "0", huffman_dict)
        get_code_for_node(node.right, code + "1", huffman_dict)


def get_huffman_dict_for_frequency(frequencies: List[Tuple[Any, int]]):
    # Convert frequencies to nodes

    nodes = []
    for item in frequencies:
        nodes.append(Node(item[0], item[1]))

    # Build Huffman tree
    while len(nodes) > 1:
        left = nodes.pop()
        right = nodes.pop()
        parent_frequency = left.frequency + right.frequency
        parent_node = Node(None, parent_frequency)
        parent_node.left = left
        parent_node.right = right
        nodes.append(parent_node)
        nodes = sorted(nodes, key=lambda x: x.frequency, reverse=True)

    # Generate Huffman codes
    huffman_dict = {}

    if nodes:
        root = nodes[0]
        get_code_for_node(root, "", huffman_dict)

    return huffman_dict

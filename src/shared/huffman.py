def huffman_dict(n):
    max_len = len(bin(n - 1)) - 2  # binary length of largest number
    huff_dict = {i: bin(i)[2:].zfill(max_len) for i in range(n)}
    return huff_dict

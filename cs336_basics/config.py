PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 1 byte = 8 bits 2^8 = 256 ==> 0 ~ 255
INIT_TOKEN = 256
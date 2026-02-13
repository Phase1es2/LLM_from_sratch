import os
import regex as re
from typing import Iterable, Optional
import json

from cs336_basics.config import PAT


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        # construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        self.vocab: dict[int, bytes] = vocab

        # fast lookup: bytes token -> id
        self.bytes_to_id: dict[bytes, int] = {b: i for i, b in vocab.items()}

        self.merges: list[tuple[bytes, bytes]] = merges

        # BPE merge rank: earlier merge has higher priority (lower rank value)
        self.merge_rank: dict[tuple[bytes, bytes], int] = {pair: idx for idx, pair in enumerate(merges)}

        self.special_tokens: list[str] = special_tokens or []

        self.RX = re.compile(PAT)

        # precompile a regex that splits & keeps special tokens as delimiters
        # Example: "hi<|endoftext|>there" -> ["hi", "<|endoftext|>", "there"]
        if self.special_tokens:
            special_sorted = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(s) for s in special_sorted]
            self.SPECIAL_SPLIT_RX = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.SPECIAL_SPLIT_RX = None


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {}

        for k, hex_bytes in raw_vocab.items():
            tok_id = int(k)
            tok_bytes = bytes.fromhex(hex_bytes)
            vocab[tok_id] = tok_bytes

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a_hex, b_hex = line.split()
                merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _bpe_merge(self, toks: list[bytes]) -> list[bytes]:
        """
        Apply bpe merge t o one token sequence (byte-level tokens)

        Standard BPE token application:
            repeat:
                pick adjacent pair with the smallest merge_rank
                merge all occurrences of that pair in the sequence
            until no pair can be merged
        """
        if len(toks) < 2:
            return toks

        while True:
            best_pair: Optional[tuple[bytes, bytes]] = None
            best_rank = None

            for i in range(len(toks) - 1):
                pair = (toks[i], toks[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            i = 0
            merged = []
            A, B = best_pair

            while i < len(toks):
                if i < len(toks) - 1 and toks[i] == A and toks[i + 1] == B:
                    merged.append(A + B)
                    i += 2
                else:
                    merged.append(toks[i])
                    i += 1
            toks = merged
            if len(toks) < 2:
                break

        return toks


    def encode(self, text: str) -> list[int]:
        # Encode an input text into a sequence of tokens IDs
        out_ids: list[int] = []

        if self.SPECIAL_SPLIT_RX is None:
            chunks = [text]
        else:
            chunks = [c for c in self.SPECIAL_SPLIT_RX.split(text) if c != ""]

        for chunk in chunks:
            if chunk in self.special_tokens:
                b_chunk = chunk.encode("utf-8")
                chunk_id = self.bytes_to_id.get(b_chunk)
                if chunk_id is None:
                    raise ValueError(f"unknown chunk {chunk}")
                out_ids.append(chunk_id)
                continue

            for m in self.RX.finditer(chunk):
                piece = m.group(0)
                if not piece:
                    continue

                b_piece = piece.encode("utf-8")
                toks: list[bytes] = [bytes([b]) for b in b_piece]
                toks = self._bpe_merge(toks)

                for t in toks:
                    t_id = self.bytes_to_id.get(t)
                    if t_id is None:
                        raise ValueError(f"unknown token {t}")
                    out_ids.append(t_id)
        return out_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for i in iterable:
            for tid in self.encode(i):
                yield tid

    def decode(self, tokens: list[int]) -> str:
        # Decode a sequence of token IDs into text.
        """
        基本 decode:
            ids -> bytes concat -> utf-8 decode (replace)
            如果 vocab是 byte-level， 直接拼接 byte 然後 decode
        """
        # self.vocab -> int to bytes
        # “牛” ==》 b'\xe7\x89\x9b'
        #    230: b'\xe7',
        #    231: b'\x89',
        #    232: b'\x9b',
        # vocab[9950] = b'\xe7' b'\x89' b'\x9b'
        b = b"".join(self.vocab[v] for v in tokens)
        return b.decode("utf-8", errors="replace")

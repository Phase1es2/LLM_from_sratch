import heapq
import os
from collections import defaultdict
from typing import Optional

import regex as re

from cs336_basics.config import PAT, INIT_TOKEN
from cs336_basics.utils.node import Node   # 建議不要用 import *，避免命名衝突
from typing import Dict, Tuple
from multiprocessing import Pool

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.utils.revpair import RevPair

# 你的 word_frequency key 型別
WordFreq = Dict[Tuple[bytes, ...], int]


def worker_count_chunk(args) -> WordFreq:
    """
    args: (input_path, start, end, special_tokens)
    每個 worker 自己 open 檔案、讀 chunk、跑 get_frequency
    """
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    text = raw.decode("utf-8", errors="ignore")

    return get_frequency(text, special_tokens)


def merge_counts(dst: WordFreq, src: WordFreq) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def parallel_get_word_frequency(
    input_path: str,
    special_tokens: list[str],
    num_processes: int,
    split_special_token: bytes = b"<|endoftext|>",
) -> WordFreq:
    # 先算 boundaries（主進程）
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_processes,
            split_special_token=split_special_token,
        )

    jobs = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    with Pool(processes=num_processes) as pool:
        partials = pool.map(worker_count_chunk, jobs, chunksize=1)

    total: WordFreq = {}
    for d in partials:
        merge_counts(total, d)

    return total



def init_vocab(vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive int")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must not be larger than 256 + len(special_tokens)")
    """
        The tokenizer vocabulary is a one-to-one mapping from bytestring token to integer ID.
        Since we're training a byte-level BPE tokenizer, our initial vocabulary is simply the set of all
        bytes. Since there are 256 possible byte values, our initial vocabulary is size 256.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(INIT_TOKEN)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    # 我們需要做幾次 merge，才能達到 vocab_size
    num_merges = vocab_size - len(vocab)
    return vocab, num_merges


def read_from_input_path(input_path: str) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()


def get_frequency(text_str: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    pre-tokenization:
      1) 用 special_tokens 切段：special token 不參與訓練統計（相當於 drop 掉）
      2) GPT-2 regex 切出 token pieces（包含前導空格、標點、換行等）
      3) piece.encode('utf-8') → bytes，再拆成單 byte token 序列
      4) 統計每個 byte-seq (tuple[bytes,...]) 出現次數
    """
    RX = re.compile(PAT)
    if not text_str:
        return {}

    text_list = [text_str]

    # 假設我們有好幾段話，和不同的special_token，我們在train時候不需要special_token
    # 我們需要把它變成一個用 special token 劃分的一句話的整體，切割時候不會跨越另外一邊去切割
    for special_token in special_tokens:
        text_corpus: list[str] = []
        for text in text_list:
            # 注意：split 會把 special_token 本身丟掉 => “不參與訓練 stats”
            text_corpus.extend(text.split(special_token))
        text_list = text_corpus

    # 接下來，我們對每一個 span 做 regex pre-tokenization，計算出現次數
    word_frequency: dict[tuple[bytes, ...], int] = {}

    for line in text_list:
        if not line:
            continue

        for m in RX.finditer(line):
            piece = m.group(0)
            if not piece:
                continue

            # piece -> utf8 bytes，再拆成單 byte token
            bts = piece.encode("utf-8")
            key = tuple(bytes([b]) for b in bts)   # e.g. " low" -> (b' ', b'l', b'o', b'w')
            word_frequency[key] = word_frequency.get(key, 0) + 1

    return word_frequency


def build_word_dll(
    words: list[tuple[tuple[bytes, ...], int]]
) -> tuple[
    list[Optional[Node]],                                  # heads[word_id]
    list[int],                                             # freqs[word_id]
    dict[tuple[bytes, bytes], int],                        # pair_count
    dict[tuple[bytes, bytes], set[tuple[int, int, Node]]], # pair_occ: (word_id, pos, left_node)
]:
    """
    把每個 unique word(byte-seq) 建成一條 doubly linked list。
    同時建立：
      - pair_count[(A,B)]：這個 pair 在所有 words 中總共出現幾次（加權 freq）
      - pair_occ[(A,B)]：這個 pair 出現在哪些位置（word_id, left_pos, left_node）
        為了 deterministic，我們把 left_node.pos 存起來，後面 merge 時可以排序。
    """
    heads: list[Optional[Node]] = [None] * len(words)
    freqs: list[int] = [0] * len(words)
    pair_cnt: dict[tuple[bytes, bytes], int] = defaultdict(int)

    # 重要：存 (word_id, pos, node) 讓迭代順序 deterministic
    pair_occ: dict[tuple[bytes, bytes], set[tuple[int, int, Node]]] = defaultdict(set)

    for word_id, (seq, freq) in enumerate(words):
        freqs[word_id] = freq
        if not seq:
            heads[word_id] = None
            continue

        prev: Optional[Node] = None
        head: Optional[Node] = None

        for j, tok in enumerate(seq):
            # Node(tok, pos=j) ：pos 用於 deterministic merge（同 word 左到右）
            cur = Node(tok, j)

            if prev is None:
                head = cur
            else:
                prev.next = cur
                cur.prev = prev

                # 建 pair stats：pair = (prev.tok, cur.tok) 出現 freq 次
                p = (prev.tok, cur.tok)
                pair_cnt[p] += freq

                # deterministic：存 word_id + prev.pos + prev node
                pair_occ[p].add((word_id, prev.pos, prev))

            prev = cur

        heads[word_id] = head

    return heads, freqs, dict(pair_cnt), pair_occ


def pre_tokenization(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = 4,
) -> tuple[
    dict[tuple[bytes, ...], int],
    list[Optional[Node]],
    list[int],
    dict[tuple[bytes, bytes], int],
    dict[tuple[bytes, bytes], set[tuple[int, int, Node]]],
]:
    split_tok = "<|endoftext|>"
    split_special_token = split_tok.encode("utf-8")
    size = os.path.getsize(input_path)
    if size < 2_000_000:
        text_str = read_from_input_path(str(input_path))

    # 找到 這個句子中，每個 pre-tokenized “word byte-seq” 出現了多少次
        word_frequency = get_frequency(text_str, special_tokens)
    else:
        word_frequency = parallel_get_word_frequency(
            input_path=str(input_path),
            special_tokens=special_tokens,
            num_processes = min(os.cpu_count() or 4, 8),
            split_special_token=split_special_token,
        )

    # 建立 DLL + pair_cnt + pair_occ（occ 帶 pos）
    words: list[tuple[tuple[bytes, ...], int]] = list(word_frequency.items())
    heads, freqs, pair_cnt, pair_occ = build_word_dll(words)

    return word_frequency, heads, freqs, pair_cnt, pair_occ


# 旧版：不再用（因為它不增量更新 pair_cnt/pair_occ，也不 deterministic）
def merge_all_occurrences(
    best_pair: tuple[bytes, bytes],
    pair_occ: dict[tuple[bytes, bytes], set[tuple[int, Node]]],
) -> int:
    A, B = best_pair
    k = 0
    occ = pair_occ.get(best_pair, set())
    for word_id, left in list(occ):
        if (not left.alive) or (left.next is None) or (not left.next.alive):
            continue
        if left.tok != A or left.next.tok != B:
            continue

        right = left.next
        left.tok = A + B
        right.alive = False

        left.next = right.next
        if right.next is not None:
            right.next.prev = left
        k += 1
    return k


def merge_best_pair(
    best: tuple[bytes, bytes],
    pair_cnt: dict[tuple[bytes, bytes], int],
    pair_occ: dict[tuple[bytes, bytes], set[tuple[int, int, Node]]],
    freqs: list[int],
    heap: list[tuple[int, RevPair]],
) -> int:
    """
    增量 merge：
      - 对 best=(A,B)，把所有出现位置合并成 (A+B)
      - 同时“局部”更新 pair_cnt 和 pair_occ（不做全量 rebuild）
      - deterministic：occurrences 按 (word_id, pos) 排序后再合并
      - pair_occ 是 lazy 的：旧的 occ 不删也没关系，下一轮会靠 alive + tok match skip
    """
    A, B = best
    def push_if_positive(p: tuple[bytes, bytes]) -> None:
        """
        把目前最新的 pair_cnt[p] 推進 heap（lazy delete）
        - 不需要把舊值從 heap 移除：之後 pop 出來時比對 pair_cnt 即可丟掉 stale
        """
        c = pair_cnt.get(p, 0)
        if c > 0:
            heapq.heappush(heap, (-c, RevPair(p)))

    occs = pair_occ.get(best, set())
    merged = 0

    # 固定顺序：word_id 小到大，同一个 word 内从左到右（pos）
    for word_id, _, left in sorted(occs, key=lambda t: (t[0], t[1])):
        if not left.alive:
            continue

        # right = 下一個 alive node（跳過被 merge 掉的 dead nodes）
        right = left.next
        while right is not None and not right.alive:
            right = right.next
        if right is None:
            continue

        # 校验一下当前是否仍然是 (A,B)
        if left.tok != A or right.tok != B:
            continue

        w = freqs[word_id]  # 这个 word 的出现次数（加权）
        x = Node.prev_alive(left)   # alive prev
        y = Node.next_alive(right)  # alive next

        # ---- decrement old pairs（把被破坏的边界 pair 从统计中减掉） ----
        pair_cnt[best] = pair_cnt.get(best, 0) - w
        push_if_positive(best)
        if x is not None:
            px = (x.tok, left.tok)
            pair_cnt[px] = pair_cnt.get(px, 0) - w
            push_if_positive(px)
        if y is not None:
            py = (right.tok, y.tok)
            pair_cnt[py] = pair_cnt.get(py, 0) - w
            push_if_positive(py)

        # ---- do merge in DLL ----
        new_tok = A + B
        left.tok = new_tok
        right.alive = False

        # relink: left 跳过 right
        left.next = right.next
        if right.next is not None:
            right.next.prev = left

        # ---- increment new boundary pairs + add occurrences（新增边界 pair） ----
        # 注意：这里也要存 (word_id, pos, node) 保持 deterministic
        if x is not None:
            p = (x.tok, left.tok)
            pair_cnt[p] = pair_cnt.get(p, 0) + w
            pair_occ[p].add((word_id, x.pos, x))
            push_if_positive(p)

        if y is not None:
            p = (left.tok, y.tok)
            pair_cnt[p] = pair_cnt.get(p, 0) + w
            pair_occ[p].add((word_id, left.pos, left))
            push_if_positive(p)

        merged += 1

    return merged


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 byte-level BPE：
      1) init vocab：0..255 + special_tokens
      2) pre-tokenization：得到 unique “byte word seq” 的频率
      3) 建 DLL + pair_cnt + pair_occ
      4) 循环做 num_merges 次：
         - 选频率最高 pair（tie 用 lexicographically greater）
         - merge + 增量更新统计
         - 记录 merges
    """
    vocab, num_merges = init_vocab(vocab_size, special_tokens)

    # pre-tokenization / build stats
    _, heads, freqs, pair_cnt, pair_occ = pre_tokenization(input_path, special_tokens)

    merges: list[tuple[bytes, bytes]] = []

    heap: list[tuple[int, RevPair]] = []
    for p, c in pair_cnt.items():
        if c > 0:
            heapq.heappush(heap, (-c, RevPair(p)))

    for _ in range(num_merges):
        # 过滤掉 <=0 的 pair（增量更新 + lazy occ 可能导致 0/负数）
        best: Optional[tuple[bytes, bytes]] = None
        while heap:
            neg_c, rp = heapq.heappop(heap)
            p = rp.p
            c = -neg_c
            cur = pair_cnt.get(p, 0)
            if cur != c or cur <= 0:
                continue
            best = p
            break
        if best is None:
            break

        A, B = best
        vocab[len(vocab)] = A + B
        merges.append(best)

        k = merge_best_pair(best, pair_cnt, pair_occ, freqs=freqs, heap=heap)
        # k==0 一般是 occ 都 stale 了（被之前 merge 影响），不致命，继续即可
        # 继续下一轮会选别的 pair

    return vocab, merges
import heapq  # 用 heap 取 best pair，避免每輪掃整張 pair_cnt（最致命瓶頸）

class RevPair:
    """
    用來處理 tie-break：你要的是「(count 相同) 時選 lexicographically greater 的 pair」。

    Python heapq 是 min-heap：
      - 我們用 (-count, something) 來做 max-heap
      - tie 時 heap 會比較第二個欄位，預設是 lexicographically smaller 先出來
      - 所以我們用 RevPair 反轉比較，讓 lexicographically greater 先出來
    """
    __slots__ = ("p",)

    def __init__(self, p: tuple[bytes, bytes]):
        self.p = p

    def __lt__(self, other: "RevPair") -> bool:
        # 反轉：greater 優先
        return self.p > other.p
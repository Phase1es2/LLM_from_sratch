from __future__ import annotations
from typing import Optional

class Node:
    __slots__ = ("tok", "prev", "next", "alive", "pos")

    def __init__(self, tok: bytes, pos: int):
        self.tok: bytes = tok
        self.pos: int = pos
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None
        self.alive: bool = True

    @staticmethod
    def prev_alive(n: Optional["Node"]) -> Optional["Node"]:
        p = n.prev if n else None
        while p is not None and not p.alive:
            p = p.prev
        return p

    @staticmethod
    def next_alive(n: Optional["Node"]) -> Optional["Node"]:
        q = n.next if n else None
        while q is not None and not q.alive:
            q = q.next
        return q
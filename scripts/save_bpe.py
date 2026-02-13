import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# -----------------------------
# Types
# -----------------------------
Vocab = Dict[int, bytes]
Merges = List[Tuple[bytes, bytes]]

# -----------------------------
# Helpers: bytes <-> hex
# -----------------------------
def b2hex(b: bytes) -> str:
    """bytes -> hex string (no 0x prefix)"""
    return b.hex()

def hex2b(s: str) -> bytes:
    """hex string -> bytes"""
    return bytes.fromhex(s)

# -----------------------------
# Longest token helper
# -----------------------------
def get_longest_token(vocab: Vocab) -> tuple[int, bytes]:
    """Return (token_id, token_bytes) of the longest token by length."""
    return max(vocab.items(), key=lambda kv: len(kv[1]))

# -----------------------------
# Save artifacts
# -----------------------------
def save_bpe_artifacts(
    out_dir: str | os.PathLike,
    vocab: Vocab,
    merges: Merges,
    *,
    special_tokens: Optional[List[str]] = None,
    input_path: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    peak_rss_mb: Optional[float] = None,
    extra_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save:
      - vocab.json      : { "id": "hexbytes" }
      - merges.txt      : each line "hexA hexB"
      - stats.json      : metadata + longest token info

    Returns stats dict (also written to disk).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) vocab.json (bytes -> hex)
    vocab_json = {str(i): b2hex(tok) for i, tok in vocab.items()}
    (out_path / "vocab.json").write_text(
        json.dumps(vocab_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2) merges.txt (bytes -> hex)
    # one merge per line: "<hexA> <hexB>"
    with (out_path / "merges.txt").open("w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{b2hex(a)} {b2hex(b)}\n")

    # 3) stats.json
    longest_id, longest_tok = get_longest_token(vocab)
    stats: Dict[str, Any] = {
        "input_path": str(input_path) if input_path is not None else None,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": special_tokens or [],
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hours": (elapsed_seconds / 3600.0) if elapsed_seconds is not None else None,
        "peak_rss_mb": peak_rss_mb,
        "peak_rss_gb": (peak_rss_mb / 1024.0) if peak_rss_mb is not None else None,
        "longest_token_id": longest_id,
        "longest_token_num_bytes": len(longest_tok),
        "longest_token_hex": b2hex(longest_tok),
    }
    if extra_stats:
        stats.update(extra_stats)

    (out_path / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # (Optional) human-readable longest token preview
    # 這份不是必需，只是方便你目視檢查 token 長什麼樣
    with (out_path / "longest_token_readable.txt").open("w", encoding="utf-8") as f:
        f.write(f"token_id: {longest_id}\n")
        f.write(f"num_bytes: {len(longest_tok)}\n")
        f.write(f"hex: {b2hex(longest_tok)}\n")
        # 用 latin-1 會 1:1 保留 bytes 值（0-255），不會 decode error
        f.write(f"latin1: {longest_tok.decode('latin-1')}\n")

    return stats

# -----------------------------
# (Optional) Load back
# -----------------------------
def load_bpe_artifacts(out_dir: str | os.PathLike) -> tuple[Vocab, Merges, Dict[str, Any]]:
    out_path = Path(out_dir)

    vocab_json = json.loads((out_path / "vocab.json").read_text(encoding="utf-8"))
    vocab: Vocab = {int(i): hex2b(h) for i, h in vocab_json.items()}

    merges: Merges = []
    with (out_path / "merges.txt").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ha, hb = line.split()
            merges.append((hex2b(ha), hex2b(hb)))

    stats = json.loads((out_path / "stats.json").read_text(encoding="utf-8"))
    return vocab, merges, stats
import os
import time

from cs336_basics.tokenizer.train_bpe import train_bpe
from scripts.save_bpe import save_bpe_artifacts


def bpe_output(input_path: str, vocab_size: int, special_tokens: list[str]) -> None:
    # 用檔名當資料夾前綴（去掉 .txt）
    base = os.path.splitext(os.path.basename(input_path))[0]

    start = time.time()
    vocab, merges = train_bpe(
        input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    elapsed = time.time() - start

    stats = save_bpe_artifacts(
        out_dir=f"{base}_bpe_{vocab_size}",
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
        input_path=input_path,
        elapsed_seconds=elapsed,
        peak_rss_mb=None,
    )

    print(f"[{base}] saved: {stats['vocab_size']} tokens, {stats['num_merges']} merges")
    print(f"[{base}] longest bytes: {stats['longest_token_num_bytes']}")


def main():
    datasets = {
        "tinystories": ("../data/TinyStoriesV2-GPT4-train.txt", 10_000),
        "owt": ("../data/owt_train.txt", 32_000),
    }

    special_tokens = ["<|endoftext|>"]

    for name, (path, vocab_size) in datasets.items():
        print(f"\n=== training {name} ===")
        bpe_output(path, vocab_size, special_tokens)


if __name__ == "__main__":
    main()
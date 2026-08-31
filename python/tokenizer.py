"""Tokenize text with Hugging Face's official Llama tokenizer.

Examples:
    python3 python/tokenizer.py "The capital of France is" \
        --model meta-llama/Llama-3.2-1B-Instruct
    python3 python/tokenizer.py --decode --ids 791 6864 315 9822 374 \
        --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode text into Llama token IDs or decode IDs into text."
    )
    parser.add_argument("text", nargs="?", help="Text to tokenize")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--decode", action="store_true", help="Decode token IDs into text")
    parser.add_argument("--ids", nargs="+", type=int, help="Token IDs to decode")
    parser.add_argument("--output", "-o", help="Optional JSON output file for encoded IDs")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.decode:
        if not args.ids:
            parser.error("--decode requires --ids")
        print(tokenizer.decode(args.ids))
        return

    if not args.text:
        parser.error("Provide text to tokenize")

    token_ids = tokenizer.encode(args.text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(token_ids, output_file)
        print(f"Wrote {len(token_ids)} tokens to {args.output}")
        return

    print(" ".join(str(token_id) for token_id in token_ids))


if __name__ == "__main__":
    main()

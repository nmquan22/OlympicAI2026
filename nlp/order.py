"""Decode a cached directed score matrix into IOAI-style chunk ranks."""
import argparse
from pathlib import Path
import numpy as np
from common.submission import order_to_ranks, write_json
from .blocks.ordering import beam_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, help="NPY [n,n], next-turn scores")
    parser.add_argument("--prefix", type=int, nargs="*", default=[])
    parser.add_argument("--dialogue-id", default="0")
    parser.add_argument("--beam", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("runs/order/answers.json"))
    args = parser.parse_args()
    scores = np.load(args.scores) if args.scores else np.array([[0, -1, 3], [0, 0, -1], [-1, 3, 0]], dtype=float)
    prefix = args.prefix if args.scores else [0, 2]
    order = beam_order(scores, prefix, args.beam)
    ranks = order_to_ranks(order)
    write_json(args.out, {args.dialogue_id: ranks})
    print({"data": "npy" if args.scores else "synthetic_fixture", "time_order": order, "ranks_by_chunk": ranks})


if __name__ == "__main__":
    main()

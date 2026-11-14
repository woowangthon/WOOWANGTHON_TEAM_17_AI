from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.dataset import ROOT, load_dataset, write_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw v2 training examples and family split integrity.")
    parser.add_argument("--path", type=Path, default=ROOT / "data" / "generated" / "preference-v2.jsonl")
    parser.add_argument("--generate", action="store_true", help="Generate the checked-in deterministic synthetic set first.")
    args = parser.parse_args()
    if args.generate:
        print(json.dumps(write_dataset(args.path), ensure_ascii=False, indent=2))
    rows = load_dataset(args.path)
    print(json.dumps({"path": str(args.path), "records": len(rows), "status": "valid"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

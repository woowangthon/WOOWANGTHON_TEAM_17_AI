from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset import import_human_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and seal human-adjudicated evaluation annotations as test-only JSONL.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "human" / "inbox.local.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "human" / "human-evaluation-v2.jsonl")
    parser.add_argument("--synthetic", type=Path, default=ROOT / "data" / "generated" / "preference-v2.jsonl",
                        help="Current synthetic JSONL used for exact and conservative character-similarity checks")
    args = parser.parse_args()
    manifest = import_human_dataset(args.input, args.output, args.synthetic)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

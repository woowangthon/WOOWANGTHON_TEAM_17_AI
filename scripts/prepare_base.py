"""Download one pinned, official Qwen base snapshot into ignored artifacts/.

Hugging Face's cache provides resumable transport. Only the copied allowlisted
snapshot files are placed under the repository artifact directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_RELATIVE = Path("artifacts/models/qwen3.5-4b-base")
ALLOWED_NAMES = {
    "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json",
    "LICENSE", "LICENSE.txt", "README.md",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_repo(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("Base artifact path must stay inside the repository")
    if resolved != resolved_root / BASE_RELATIVE:
        raise ValueError("Base artifact destination is fixed to artifacts/models/qwen3.5-4b-base")
    return resolved


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _snapshot_files(model_info: Any) -> dict[str, dict[str, Any]]:
    siblings = _field(model_info, "siblings", [])
    result: dict[str, dict[str, Any]] = {}
    for sibling in siblings:
        name = _field(sibling, "rfilename")
        if not isinstance(name, str) or "/" in name or "\\" in name:
            continue
        lfs = _field(sibling, "lfs")
        oid = _field(lfs, "oid") if lfs else None
        size = _field(sibling, "size")
        lfs_size = _field(lfs, "size") if lfs else None
        if lfs and not isinstance(size, int):
            size = lfs_size
        result[name] = {"size": size, "lfsSha256": oid, "lfsSize": lfs_size}
    return result


def _allowed(name: str) -> bool:
    return name in ALLOWED_NAMES or name.endswith(".safetensors")


def _read_weight_shards(index_path: Path) -> set[str]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Official safetensors index has no weight_map")
    shards = set(weight_map.values())
    if any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors") for name in shards):
        raise ValueError("Unsafe weight shard path in model index")
    return shards


def create_manifest(base_path: Path, model_info: Any) -> dict[str, Any]:
    official = _snapshot_files(model_info)
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in base_path.rglob("*") if item.is_file()):
        relative = path.relative_to(base_path).as_posix()
        if Path(relative).name not in official or not _allowed(Path(relative).name):
            raise ValueError(f"Unexpected file in base snapshot: {relative}")
        expected = official[Path(relative).name]
        size = path.stat().st_size
        if not isinstance(expected["size"], int) or size != expected["size"]:
            raise ValueError(f"Base file size differs from official metadata: {relative}")
        checksum = sha256_file(path)
        official_lfs_sha = expected["lfsSha256"]
        if official_lfs_sha is not None:
            lfs_size = expected["lfsSize"]
            if not isinstance(lfs_size, int) or size != lfs_size:
                raise ValueError(f"Base file differs from official LFS size: {relative}")
            if not SHA256.fullmatch(str(official_lfs_sha)) or checksum != official_lfs_sha or size <= 0:
                raise ValueError(f"Base file differs from official LFS metadata: {relative}")
        record = {"file": relative, "bytes": size, "sha256": checksum,
                  "officialLfsVerified": official_lfs_sha is not None}
        if official_lfs_sha is not None:
            record["officialLfsSha256"] = official_lfs_sha
            record["officialLfsBytes"] = expected["lfsSize"]
        files.append(record)
    names = {entry["file"] for entry in files}
    required = {"config.json", "tokenizer.json", "model.safetensors.index.json"}
    if not required <= names:
        raise ValueError("Official snapshot is missing required model/tokenizer files")
    shards = _read_weight_shards(base_path / "model.safetensors.index.json")
    if not shards <= names:
        raise ValueError("Official snapshot is missing a referenced safetensors shard")
    return {"repository": MODEL_ID, "revision": REVISION,
            "verifiedAt": datetime.now(timezone.utc).isoformat(), "files": files}


def prepare_base(root: Path = ROOT, *, api: Any = None, downloader: Any = None) -> dict[str, Any]:
    root = root.resolve()
    models_dir = (root / "artifacts" / "models").resolve()
    target = _inside_repo(root / BASE_RELATIVE, root)
    manifest_path = models_dir / "base-manifest.json"
    download_path = models_dir / "base-download.json"
    if target.exists() or manifest_path.exists():
        if not target.is_dir() or not manifest_path.is_file():
            raise FileExistsError("Base destination exists without a complete verified manifest; refusing to overwrite")
        from serving.verification import verify_base_snapshot
        result = verify_base_snapshot(target, MODEL_ID, REVISION)
        return {"status": "already_verified", **result}

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install the optional huggingface_hub package to download the pinned base") from exc
    api = api or HfApi()
    downloader = downloader or hf_hub_download
    info = api.model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if _field(info, "sha") != REVISION:
        raise ValueError("Hugging Face resolved a different revision than the pinned commit")
    official = _snapshot_files(info)
    names = {name for name in official if _allowed(name) and not name.endswith(".safetensors")}
    required = {"config.json", "tokenizer.json", "model.safetensors.index.json"}
    if not required <= names:
        raise ValueError("Pinned repository metadata omits required files")
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qwen-base-", dir=models_dir) as temporary:
        staging = Path(temporary) / BASE_RELATIVE.name
        staging.mkdir()
        for name in sorted(names):
            cached = Path(downloader(repo_id=MODEL_ID, filename=name, revision=REVISION))
            shutil.copyfile(cached, staging / name)
        shards = _read_weight_shards(staging / "model.safetensors.index.json")
        if not shards <= official.keys() or any(not _allowed(name) for name in shards):
            raise ValueError("The model index references missing or disallowed weight shards")
        for name in sorted(shards - names):
            cached = Path(downloader(repo_id=MODEL_ID, filename=name, revision=REVISION))
            shutil.copyfile(cached, staging / name)
        manifest = create_manifest(staging, info)
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        (Path(temporary) / "base-manifest.json").write_bytes(manifest_bytes)
        marker = {"repository": MODEL_ID, "revision": REVISION, "status": "downloaded",
                  "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "downloadedFiles": len(manifest["files"]),
                  "totalBytes": sum(file["bytes"] for file in manifest["files"]),
                  "verifiedAt": manifest["verifiedAt"]}
        marker_bytes = (json.dumps(marker, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        (Path(temporary) / "base-download.json").write_bytes(marker_bytes)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(target)
        (Path(temporary) / "base-manifest.json").replace(manifest_path)
        (Path(temporary) / "base-download.json").replace(download_path)
    from serving.verification import verify_base_snapshot
    result = verify_base_snapshot(target, MODEL_ID, REVISION)
    return {"status": "downloaded", **result, "files": len(manifest["files"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(json.dumps(prepare_base(args.root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

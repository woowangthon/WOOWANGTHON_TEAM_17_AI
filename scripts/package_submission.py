"""Create a checksummed, path-safe handoff archive for a trained adapter."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "submission.json"
WINDOWS_USER_PATH = re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/]+", re.IGNORECASE)
UNIX_USER_PATH = re.compile(r"/(?:home|Users)/[^/\s]+|/root(?:/|\b)")
FORBIDDEN_PARTS = {".env", ".cache", "cache", "__pycache__", ".venv", "venv", "node_modules", ".git"}
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}


class SubmissionError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_file_manifest(entries: dict[str, bytes]) -> list[dict[str, Any]]:
    return [{"path": safe_archive_name(name), "bytes": len(data), "sha256": sha256_bytes(data)}
            for name, data in sorted(entries.items())]


def safe_relative(root: Path, value: str | Path, *, required: bool = True) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise SubmissionError(f"Submission paths must be relative to the repository: {value}")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise SubmissionError("Submission path escapes the repository")
    if required and not resolved.exists():
        raise SubmissionError(f"Required submission input is missing: {path.as_posix()}")
    return resolved


def safe_archive_name(value: str) -> str:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or ":" in pure.parts[0]:
        raise SubmissionError(f"Unsafe archive name: {value}")
    if any(part.lower() in FORBIDDEN_PARTS or part.lower().startswith(".env.") for part in pure.parts):
        raise SubmissionError(f"Private/cache path cannot be packaged: {value}")
    if pure.suffix.lower() in WEIGHT_SUFFIXES:
        adapter_weights = {"adapter_model.safetensors", "adapter_model.bin"}
        if pure.parts[0] != "adapter" or pure.name not in adapter_weights:
            raise SubmissionError(f"Base/model weight files cannot be packaged: {value}")
    return pure.as_posix()


def _redact_root(value: str, root: Path) -> tuple[str, bool]:
    variants = {str(root), str(root).replace("\\", "/"), str(root).replace("/", "\\")}
    changed = False
    for variant in sorted(variants, key=len, reverse=True):
        if variant and variant in value:
            value = value.replace(variant, ".")
            changed = True
    if WINDOWS_USER_PATH.search(value) or UNIX_USER_PATH.search(value):
        raise SubmissionError("A local user path remains in a packaged text artifact")
    return value, changed


def _sanitize_json_value(value: Any, root: Path, *, adapter_config: bool = False) -> tuple[Any, bool]:
    changed = False
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if adapter_config and key == "base_model_name_or_path":
                result[key] = "Qwen/Qwen3.5-4B"
                changed = changed or item != result[key]
            else:
                result[key], item_changed = _sanitize_json_value(item, root)
                changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        result = []
        for item in value:
            clean, item_changed = _sanitize_json_value(item, root)
            result.append(clean)
            changed = changed or item_changed
        return result, changed
    if isinstance(value, str):
        clean, changed = _redact_root(value, root)
        return clean, changed
    return value, False


def sanitize_text(data: bytes, root: Path, name: str) -> tuple[bytes, bool]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data, False
    if name.endswith(".json"):
        parsed = json.loads(text)
        sanitized, changed = _sanitize_json_value(parsed, root, adapter_config=Path(name).name == "adapter_config.json")
        return (json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n").encode("utf-8"), changed
    if name.endswith(".jsonl"):
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
        cleaned = []
        changed = False
        for item in values:
            safe, item_changed = _sanitize_json_value(item, root)
            cleaned.append(json.dumps(safe, ensure_ascii=False, separators=(",", ":")))
            changed = changed or item_changed
        return ("\n".join(cleaned) + "\n").encode("utf-8"), changed
    clean, changed = _redact_root(text, root)
    return clean.encode("utf-8"), changed


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return result if re.fullmatch(r"[0-9a-f]{40,64}", result) else None


def _source_repository(root: Path, label: str, value: str) -> Path:
    candidate = (root / value).resolve()
    expected_name = {"AI": root.name, "FE": "ensemble-FE", "BE": "ensemble-BE"}.get(label)
    if expected_name is None or candidate.name != expected_name:
        raise SubmissionError(f"Unexpected {label} source repository reference")
    if label == "AI" and candidate != root.resolve():
        raise SubmissionError("AI source repository must be this repository")
    if label in {"FE", "BE"} and candidate.parent != root.resolve().parent:
        raise SubmissionError("FE/BE source repositories must be sibling repositories")
    return candidate


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionError("Submission config is missing or invalid") from exc
    if not isinstance(value, dict):
        raise SubmissionError("Submission config must be a JSON object")
    return value


def _human_provenance(value: Any) -> bool:
    return isinstance(value, str) and ("human" in value.casefold() or "adjudicated" in value.casefold())


def _synthetic_provenance(value: Any) -> bool:
    return isinstance(value, str) and value.casefold().startswith("synthetic_")


def _report_provenance(document: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Return (human present, synthetic explicitly identified, structure valid)."""
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        return False, False, False
    if "containsHumanData" in metadata and type(metadata["containsHumanData"]) is not bool:
        return False, False, False
    human = metadata.get("containsHumanData") is True
    labels: list[str] = []
    by_provenance = document.get("byProvenance")
    if by_provenance is not None:
        if not isinstance(by_provenance, dict):
            return False, False, False
        for label, summary in by_provenance.items():
            if not isinstance(label, str) or not isinstance(summary, dict):
                return False, False, False
            examples = summary.get("examples")
            if examples is not None and (type(examples) is not int or examples < 0):
                return False, False, False
            if examples == 0:
                continue
            labels.append(label)
            human = human or _human_provenance(label)
    examples = document.get("examples")
    if examples is not None:
        if not isinstance(examples, list) or any(not isinstance(row, dict) for row in examples):
            return False, False, False
        for row in examples:
            provenance = row.get("provenance", row.get("labelStatus"))
            if not isinstance(provenance, str):
                labels.append("unknown")
                continue
            labels.append(provenance)
            human = human or _human_provenance(provenance)
    declared_provenance = metadata.get("provenance")
    if isinstance(declared_provenance, str):
        labels.append(declared_provenance)
        human = human or _human_provenance(declared_provenance)
    elif isinstance(declared_provenance, list):
        if any(not isinstance(label, str) for label in declared_provenance):
            return False, False, False
        labels.extend(declared_provenance)
        human = human or any(_human_provenance(label) for label in declared_provenance)
    elif declared_provenance is not None:
        return False, False, False
    synthetic_explicit = bool(labels) and all(_synthetic_provenance(label) for label in labels)
    return human, synthetic_explicit, True


def _evaluation_report_eligibility(json_path: Path) -> str | None:
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "malformed_report_json"
    if not isinstance(document, dict):
        return "malformed_report_json"
    human, synthetic_explicit, valid = _report_provenance(document)
    if not valid:
        return "malformed_or_unknown_privacy_metadata"
    metadata = document["metadata"]
    permission = metadata.get("publicRedistributionAllowed")
    if permission is not None and type(permission) is not bool:
        return "malformed_or_unknown_privacy_metadata"
    if permission is False:
        return "public_redistribution_denied"
    if human:
        if permission is not True:
            return "human_report_permission_missing"
        return None
    if synthetic_explicit:
        return None
    if metadata.get("containsHumanData") is False and permission is True:
        return None
    return "unknown_privacy_provenance"


def _csv_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _valid_csv_report(path: Path, document: dict[str, Any]) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            actual = list(reader)
            examples = document.get("examples")
            if not isinstance(examples, list) or not examples or any(not isinstance(row, dict) for row in examples):
                return False
            expected_header = sorted({key for row in examples for key in row})
            if not expected_header or not actual or actual[0] != expected_header:
                return False
            expected_rows = [[_csv_value(row.get(key)) for key in expected_header] for row in examples]
            return actual[1:] == expected_rows
    except (OSError, UnicodeError, csv.Error):
        return False


def collect_evaluation_reports(directory: Path, root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Return only eligible, well-formed JSON/CSV pairs with safe exclusion reasons."""
    if not directory.is_dir():
        return [], []
    json_by_stem = {path.stem: path for path in directory.glob("*") if path.is_file() and path.suffix.lower() == ".json"}
    csv_by_stem = {path.stem: path for path in directory.glob("*") if path.is_file() and path.suffix.lower() == ".csv"}
    included: list[Path] = []
    excluded: list[dict[str, str]] = []
    for stem in sorted(set(json_by_stem) | set(csv_by_stem)):
        json_path = json_by_stem.get(stem)
        csv_path = csv_by_stem.get(stem)
        pair = [path for path in (json_path, csv_path) if path is not None]
        if json_path is None or csv_path is None:
            reason = "missing_json_pair" if json_path is None else "unpaired_json_report"
        else:
            reason = _evaluation_report_eligibility(json_path)
            if reason is None:
                try:
                    document = json.loads(json_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    document = None
                if not isinstance(document, dict) or not _valid_csv_report(csv_path, document):
                    reason = "csv_does_not_match_json_examples"
        if reason:
            for path in pair:
                excluded.append({"name": path.relative_to(root).as_posix(), "reason": reason})
            continue
        included.extend(pair)
    return included, excluded


def _evaluation_status(included: list[Path], excluded: list[dict[str, str]]) -> str:
    if included and excluded:
        return "reports_available_with_exclusions"
    if included:
        return "reports_available"
    if excluded:
        return "reports_excluded_by_privacy_or_pairing"
    return "not_run_or_not_available"


def _verify_dataset(path: Path) -> Path:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise SubmissionError("Dataset manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(path) != manifest.get("sha256"):
        raise SubmissionError("Dataset bytes do not match the dataset manifest")
    return manifest_path


def _add_file(entries: dict[str, bytes], name: str, path: Path, root: Path, sanitized: list[str]) -> None:
    relative = safe_archive_name(name)
    if path.is_symlink() or not path.is_file():
        raise SubmissionError(f"Not a regular file: {name}")
    data, changed = sanitize_text(path.read_bytes(), root, relative)
    entries[relative] = data
    if changed:
        sanitized.append(relative)


def _adapter_entries(root: Path, adapter_path: Path, expected_model: str, expected_revision: str,
                     entries: dict[str, bytes], sanitized: list[str]) -> dict[str, Any]:
    from serving.verification import verify_adapter_manifest
    manifest_path = adapter_path / "adapter-manifest.json"
    if not manifest_path.is_file():
        raise SubmissionError("Adapter provenance manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("status")
    if status == "smoke-only":
        raise SubmissionError("Smoke-only adapter cannot be submitted")
    if status not in {"trained-not-reviewed", "trained-reviewed"}:
        raise SubmissionError("Adapter status is not an accepted trained artifact status")
    original_verification = verify_adapter_manifest(adapter_path, expected_model, expected_revision)
    adapter_prefix = "adapter"
    for file in sorted(adapter_path.rglob("*")):
        if not file.is_file() or file.is_symlink():
            continue
        relative = file.relative_to(adapter_path).as_posix()
        if relative == "adapter-manifest.json":
            continue
        _add_file(entries, f"{adapter_prefix}/{relative}", file, root, sanitized)
    packaged_file_hashes = {
        name.removeprefix(adapter_prefix + "/"): sha256_bytes(data)
        for name, data in entries.items() if name.startswith(adapter_prefix + "/")
    }
    manifest_copy = dict(manifest)
    manifest_copy["adapterFilesSha256"] = packaged_file_hashes
    manifest_copy["packagedFromManifestSha256"] = original_verification["manifestSha256"]
    packaged_manifest_bytes = (json.dumps(manifest_copy, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    entries[f"{adapter_prefix}/adapter-manifest.json"] = packaged_manifest_bytes
    return {"status": status, "originalManifestSha256": original_verification["manifestSha256"],
            "packagedManifestSha256": sha256_bytes(packaged_manifest_bytes),
            "contractHash": original_verification["contractHash"],
            "systemPromptSha256": original_verification["systemPromptSha256"],
            "adapterFiles": len(packaged_file_hashes)}


def inspect_submission(root: Path = ROOT, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read-only dry run. It reports missing inputs instead of implying a package exists."""
    root = root.resolve()
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = _load_config(config_path)
    adapter = safe_relative(root, config["adapter_path"], required=False)
    errors: list[str] = []
    if not adapter.is_dir():
        errors.append(f"Trained adapter is missing: {Path(config['adapter_path']).as_posix()}; run the full training config first")
    elif not (adapter / "adapter-manifest.json").is_file():
        errors.append("Adapter provenance manifest is missing")
    elif json.loads((adapter / "adapter-manifest.json").read_text(encoding="utf-8")).get("status") == "smoke-only":
        errors.append("Smoke-only adapter cannot be submitted")
    eval_dir = safe_relative(root, config["evaluation_directory"], required=False)
    eval_reports, excluded_reports = collect_evaluation_reports(eval_dir, root)
    eval_report_names = sorted(path.relative_to(root).as_posix() for path in eval_reports)
    return {"packageStatus": "not_packaged" if errors else "ready_to_validate",
            "errors": errors, "adapterPath": Path(config["adapter_path"]).as_posix(),
            "evaluationStatus": _evaluation_status(eval_reports, excluded_reports),
            "evaluationReports": eval_report_names,
            "excludedEvaluationReports": excluded_reports,
            "humanGoldStatus": "pending_team_input"}


def create_submission(root: Path = ROOT, config_path: Path = DEFAULT_CONFIG, *, output_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = _load_config(config_path)
    model_id = config.get("model_id")
    revision = config.get("base_revision")
    if model_id != "Qwen/Qwen3.5-4B" or revision != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a":
        raise SubmissionError("Submission config must use the pinned official base model and immutable revision")
    adapter_path = safe_relative(root, config["adapter_path"], required=False)
    if not adapter_path.is_dir():
        raise SubmissionError(f"Trained adapter is missing: {Path(config['adapter_path']).as_posix()}; run the full training config first")
    adapter_manifest_path = adapter_path / "adapter-manifest.json"
    if not adapter_manifest_path.is_file():
        raise SubmissionError("Adapter provenance manifest is missing")
    source_adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    if source_adapter_manifest.get("status") == "smoke-only":
        raise SubmissionError("Smoke-only adapter cannot be submitted")
    base_path = safe_relative(root, config["base_path"])
    from serving.verification import verify_base_snapshot
    base_verification = verify_base_snapshot(base_path, model_id, revision)
    base_manifest_path = base_path.parent / "base-manifest.json"
    base_download_path = base_path.parent / "base-download.json"
    license_path = base_path / "LICENSE"
    if not license_path.is_file():
        raise SubmissionError("Official base model license notice is missing")
    dataset = safe_relative(root, config["dataset"])
    dataset_manifest = _verify_dataset(dataset)
    training_run = safe_relative(root, config["training_run"])
    train_log = training_run / "training.jsonl"
    if not train_log.is_file():
        raise SubmissionError("Full training log is missing")
    train_config = safe_relative(root, config["training_config"])
    eval_config = safe_relative(root, config["evaluation_config"])
    required_docs = ["TEAM_INPUTS.md", "docs/reproducibility.md", "docs/submission.md"]
    for name in required_docs:
        safe_relative(root, name)
    output_relative = Path(config["output"])
    output_dir = safe_relative(root, output_relative, required=False)
    if output_path is None:
        output_path = output_dir / "qwen3.5-4b-submission.zip"
    elif not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    if not output_path.is_relative_to((root / "artifacts" / "submission").resolve()):
        raise SubmissionError("Submission archive output must stay inside artifacts/submission")

    entries: dict[str, bytes] = {}
    sanitized: list[str] = []
    adapter_summary = _adapter_entries(root, adapter_path, model_id, revision, entries, sanitized)
    from serving.verification import verify_adapter_manifest
    # Verify the packaged adapter after path-safe metadata normalization.
    package_temp = root / "artifacts" / "submission"
    package_temp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="submission-check-", dir=package_temp) as temporary:
        temp_root = Path(temporary)
        staged_adapter = temp_root / "adapter"
        for name, data in entries.items():
            if name.startswith("adapter/"):
                target = staged_adapter / name.removeprefix("adapter/")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        verify_adapter_manifest(staged_adapter, model_id, revision)

    sources: dict[str, str | None] = {}
    source_paths = config.get("source_repositories", {})
    for label, relative in source_paths.items():
        path = _source_repository(root, label, relative)
        sources[label] = _git_commit(path) if path.is_dir() else None

    for archive_name, path in [
        ("provenance/base-manifest.json", base_manifest_path),
        ("provenance/base-download.json", base_download_path),
        ("notices/Qwen-Qwen3.5-4B/LICENSE", license_path),
        ("training/training.jsonl", train_log),
        ("training/config.json", train_config),
        ("training/adapter-manifest.json", training_run / "adapter-manifest.json"),
        ("data/preference-v2.jsonl", dataset),
        ("data/preference-v2.manifest.json", dataset_manifest),
        ("evaluation/config.json", eval_config),
        ("team/TEAM_INPUTS.md", root / "TEAM_INPUTS.md"),
        ("reproduction/reproducibility.md", root / "docs" / "reproducibility.md"),
        ("reproduction/submission.md", root / "docs" / "submission.md"),
        ("reproduction/requirements-training.txt", root / "requirements-training.txt"),
        ("reproduction/requirements-serving.txt", root / "requirements-serving.txt"),
        ("reproduction/requirements-kernels.txt", root / "requirements-kernels.txt"),
        ("reproduction/requirements-lock.windows.txt", root / "requirements-lock.windows.txt"),
        ("evaluation/evaluation-local.json", root / "configs" / "evaluation-local.json"),
        ("evaluation/evaluation-gemini.json", root / "configs" / "evaluation-gemini.json"),
        ("evaluation/evaluation-ensemble.json", root / "configs" / "evaluation-ensemble.json"),
        ("evidence/RESULTS.md", root / "reports" / "RESULTS.md"),
        ("evidence/EVALUATION_SUMMARY.json", root / "reports" / "EVALUATION_SUMMARY.json"),
        ("evidence/PRESENTATION_NOTES.md", root / "reports" / "PRESENTATION_NOTES.md"),
        ("evidence/ui/README.md", root / "reports" / "ui" / "README.md"),
        ("evidence/ui/preference-v2-desktop.jpg", root / "reports" / "ui" / "preference-v2-desktop.jpg"),
        ("evidence/ui/preference-v2-mobile.jpg", root / "reports" / "ui" / "preference-v2-mobile.jpg"),
    ]:
        if path.is_file():
            _add_file(entries, archive_name, path, root, sanitized)
    evaluation_dir = safe_relative(root, config["evaluation_directory"], required=False)
    selected_reports, excluded_reports = collect_evaluation_reports(evaluation_dir, root)
    evaluation_reports: list[str] = []
    for report in selected_reports:
        name = f"evaluation/reports/{report.name}"
        _add_file(entries, name, report, root, sanitized)
        evaluation_reports.append(name)
    entries["provenance/source-commits.json"] = (json.dumps(sources, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    package_metadata = {
        "formatVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "packageStatus": "artifact_handoff_not_release_approval",
        "baseModel": model_id,
        "baseRevision": revision,
        "baseManifestSha256": base_verification["manifestSha256"],
        "adapter": adapter_summary,
        "sourceCommits": sources,
        "evaluationStatus": _evaluation_status(selected_reports, excluded_reports),
        "evaluationReports": evaluation_reports,
        "excludedEvaluationReports": excluded_reports,
        "humanGoldStatus": "pending_team_input",
        "teamInputsIncluded": True,
        "sanitizedTextPaths": sorted(set(sanitized)),
        "fileManifestPath": "provenance/submission-files.json",
        "fileManifestExcludesItself": True,
    }
    entries["provenance/submission-metadata.json"] = (json.dumps(package_metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    files_manifest = create_file_manifest(entries)
    entries["provenance/submission-files.json"] = (json.dumps(files_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for name in entries:
        safe_archive_name(name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = output_path.with_suffix(output_path.suffix + ".tmp")
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    temporary_zip.replace(output_path)
    checksum = sha256_file(output_path)
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {output_path.name}\n", encoding="ascii")
    return {"status": "packaged", "archive": output_path.relative_to(root).as_posix(),
            "archiveSha256": checksum, "checksumFile": checksum_path.relative_to(root).as_posix(),
            "files": len(entries), "adapterStatus": adapter_summary["status"],
            "evaluationStatus": package_metadata["evaluationStatus"],
            "excludedEvaluationReports": excluded_reports,
            "humanGoldStatus": package_metadata["humanGoldStatus"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Report missing inputs and status without creating files")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.dry_run:
            result = inspect_submission(args.root, args.config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if not result["errors"] else 2
        result = create_submission(args.root, args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (SubmissionError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "not_packaged", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

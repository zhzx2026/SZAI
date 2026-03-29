from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LANGUAGES = ["C++", "Python", "Java"]
DEFAULT_LICENSES = [
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "ISC",
    "MPL-2.0",
]
CURATION_MARKERS = {
    "awesome",
    "awesome-list",
    "curated-list",
    "roadmap",
    "interview",
    "cheatsheet",
    "cheat-sheet",
    "tutorial",
    "university",
}
LANGUAGE_ALIASES = {
    "c++": "C++",
    "cpp": "C++",
    "python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "go": "Go",
    "rust": "Rust",
    "c#": "C#",
    "csharp": "C#",
    "kotlin": "Kotlin",
    "swift": "Swift",
}
STRICT_LANGUAGE_EXTENSIONS = {
    "C++": {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp", ".tpp"},
    "Python": {".py"},
    "Java": {".java"},
    "JavaScript": {".js", ".jsx"},
    "TypeScript": {".ts", ".tsx"},
    "Go": {".go"},
    "Rust": {".rs"},
    "C#": {".cs"},
    "Kotlin": {".kt", ".kts"},
    "Swift": {".swift"},
}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".ipp",
    ".inl",
    ".tpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".lua",
    ".sql",
    ".proto",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
}
CODE_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "CMakeLists.txt",
    "Rakefile",
    "Gemfile",
    "Vagrantfile",
    "Jenkinsfile",
}
EXCLUDED_DIRS = {
    ".git",
    ".github",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "out",
    "target",
    "coverage",
    "docs",
    "doc",
    "third_party",
    "deps",
    "tmp",
    "temp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a code-training corpus from top-starred GitHub repositories.")
    parser.add_argument("--output-dir", default="build/code-corpus", help="Directory for generated corpus files.")
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated languages to search on GitHub.",
    )
    parser.add_argument("--repo-limit", type=int, default=18, help="Maximum number of repositories to keep.")
    parser.add_argument("--per-language-limit", type=int, default=6, help="Maximum number of repositories per language.")
    parser.add_argument("--min-stars", type=int, default=10000, help="Minimum star threshold in GitHub Search.")
    parser.add_argument("--max-repo-size-kb", type=int, default=80000, help="Maximum repo size accepted from search.")
    parser.add_argument(
        "--license-allowlist",
        default=",".join(DEFAULT_LICENSES),
        help="Comma-separated SPDX licenses to allow. Empty means no license filtering.",
    )
    parser.add_argument("--max-files-per-repo", type=int, default=40, help="Maximum extracted files per repository.")
    parser.add_argument("--min-code-files-per-repo", type=int, default=10, help="Minimum extracted files required.")
    parser.add_argument("--max-bytes-per-file", type=int, default=16000, help="Skip files larger than this many bytes.")
    parser.add_argument("--max-bytes-per-repo", type=int, default=200000, help="Maximum bytes contributed by one repo.")
    parser.add_argument("--max-total-files", type=int, default=480, help="Maximum extracted files overall.")
    parser.add_argument("--max-total-bytes", type=int, default=1800000, help="Maximum total corpus size in bytes.")
    parser.add_argument("--clone-depth", type=int, default=1, help="Git clone depth.")
    parser.add_argument(
        "--strict-language-files",
        action="store_true",
        help="Only keep source files that match the selected languages.",
    )
    return parser.parse_args()


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_languages(languages: List[str]) -> List[str]:
    normalized = []
    seen = set()
    for language in languages:
        key = language.strip()
        if not key:
            continue
        normalized_language = LANGUAGE_ALIASES.get(key.lower(), key)
        if normalized_language in seen:
            continue
        seen.add(normalized_language)
        normalized.append(normalized_language)
    return normalized


def build_allowed_extensions(languages: List[str], strict_language_files: bool) -> set[str] | None:
    if not strict_language_files:
        return None

    allowed_extensions = set()
    for language in languages:
        allowed_extensions.update(STRICT_LANGUAGE_EXTENSIONS.get(language, set()))
    return allowed_extensions


def github_api_get_json(url: str, token: str | None) -> Dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SZ-AI-Code-Corpus",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def search_repositories(
    language: str,
    min_stars: int,
    max_repo_size_kb: int,
    token: str | None,
    per_page: int,
) -> List[Dict]:
    query = (
        f"stars:>={min_stars} archived:false fork:false mirror:false "
        f"size:<={max_repo_size_kb} language:{language}"
    )
    url = (
        "https://api.github.com/search/repositories?q="
        + urllib.parse.quote_plus(query)
        + f"&sort=stars&order=desc&per_page={per_page}"
    )
    payload = github_api_get_json(url, token)
    return payload.get("items", [])


def should_skip_repo(repo: Dict, allow_licenses: set[str]) -> str | None:
    if repo.get("disabled") or repo.get("archived") or repo.get("fork"):
        return "repo-state"
    lowered_name = (repo.get("name") or "").lower()
    lowered_description = (repo.get("description") or "").lower()
    topics = {topic.lower() for topic in repo.get("topics") or []}
    if (
        lowered_name == "awesome"
        or lowered_name.startswith("awesome-")
        or "curated list" in lowered_description
        or "awesome list" in lowered_description
        or topics.intersection(CURATION_MARKERS)
    ):
        return "curation-repo"
    license_info = repo.get("license")
    spdx = license_info.get("spdx_id") if license_info else None
    if allow_licenses and spdx not in allow_licenses:
        return f"license:{spdx or 'missing'}"
    return None


def is_probably_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:1024]
    except OSError:
        return True
    return b"\x00" in chunk


def should_keep_file(path: Path, max_bytes_per_file: int, allowed_extensions: set[str] | None) -> bool:
    if path.name.startswith("."):
        return False
    if path.name.endswith(".min.js"):
        return False
    normalized_suffix = path.suffix.lower()
    if allowed_extensions is None:
        if normalized_suffix not in CODE_EXTENSIONS and path.name not in CODE_FILENAMES:
            return False
    else:
        if normalized_suffix not in allowed_extensions:
            return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size == 0 or size > max_bytes_per_file:
        return False
    if is_probably_binary(path):
        return False
    return True


def iter_code_files(repo_dir: Path, max_bytes_per_file: int, allowed_extensions: set[str] | None) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [directory for directory in dirs if directory not in EXCLUDED_DIRS and not directory.startswith(".")]
        root_path = Path(root)
        for filename in sorted(files):
            path = root_path / filename
            if should_keep_file(path, max_bytes_per_file=max_bytes_per_file, allowed_extensions=allowed_extensions):
                yield path


def clone_repository(clone_url: str, target_dir: Path, clone_depth: int) -> None:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth",
            str(clone_depth),
            "--single-branch",
            clone_url,
            str(target_dir),
        ],
        check=True,
        env=env,
    )


def slugify_repo_name(full_name: str) -> str:
    return full_name.replace("/", "__")


def extract_repository_corpus(
    repo: Dict,
    repo_dir: Path,
    repo_output_dir: Path,
    max_files_per_repo: int,
    min_code_files_per_repo: int,
    max_bytes_per_file: int,
    max_bytes_per_repo: int,
    remaining_files: int,
    remaining_bytes: int,
    allowed_extensions: set[str] | None,
) -> Dict:
    kept_files: List[Dict[str, str | int]] = []
    repo_bytes = 0
    repo_path = repo_output_dir / f"{slugify_repo_name(repo['full_name'])}.txt"

    with repo_path.open("w", encoding="utf-8") as handle:
        for path in iter_code_files(
            repo_dir,
            max_bytes_per_file=max_bytes_per_file,
            allowed_extensions=allowed_extensions,
        ):
            if len(kept_files) >= max_files_per_repo or len(kept_files) >= remaining_files:
                break
            if repo_bytes >= max_bytes_per_repo or repo_bytes >= remaining_bytes:
                break

            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            remaining_repo_budget = min(max_bytes_per_repo - repo_bytes, remaining_bytes - repo_bytes)
            snippet = content[:remaining_repo_budget]
            relative_path = path.relative_to(repo_dir).as_posix()

            record = (
                f"<repo name=\"{repo['full_name']}\" language=\"{repo.get('language') or 'Unknown'}\" "
                f"license=\"{repo.get('license_spdx') or 'UNKNOWN'}\" stars=\"{repo['stargazers_count']}\">\n"
                f"<file path=\"{relative_path}\">\n"
                f"{snippet}\n"
                f"</file>\n"
                f"</repo>\n\n"
            )
            encoded = record.encode("utf-8")
            if len(encoded) > remaining_repo_budget:
                continue

            handle.write(record)
            repo_bytes += len(encoded)
            kept_files.append({"path": relative_path, "bytes": len(encoded)})

    if len(kept_files) < min_code_files_per_repo:
        repo_path.unlink(missing_ok=True)
        return {
            "accepted": False,
            "reason": f"too-few-code-files:{len(kept_files)}",
            "repo_file": None,
            "files": kept_files,
            "bytes": repo_bytes,
        }

    return {
        "accepted": True,
        "reason": None,
        "repo_file": str(repo_path),
        "files": kept_files,
        "bytes": repo_bytes,
    }


def main() -> None:
    global args
    args = parse_args()

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repos_output_dir = output_dir / "repos"
    repos_output_dir.mkdir(parents=True, exist_ok=True)

    languages = normalize_languages(split_csv(args.languages))
    allow_licenses = set(split_csv(args.license_allowlist))
    allowed_extensions = build_allowed_extensions(languages, strict_language_files=args.strict_language_files)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    selected_repos = []
    rejected_repos = []
    seen = set()
    remaining_total_files = args.max_total_files
    remaining_total_bytes = args.max_total_bytes

    with tempfile.TemporaryDirectory(prefix="sz-ai-code-corpus-") as temp_dir_name:
        temp_root = Path(temp_dir_name)

        for language in languages:
            if len(selected_repos) >= args.repo_limit or remaining_total_files <= 0 or remaining_total_bytes <= 0:
                break

            language_kept = 0
            candidates = search_repositories(
                language=language,
                min_stars=args.min_stars,
                max_repo_size_kb=args.max_repo_size_kb,
                token=token,
                per_page=max(args.per_language_limit * 8, 20),
            )

            for repo in candidates:
                if language_kept >= args.per_language_limit or len(selected_repos) >= args.repo_limit:
                    break
                if remaining_total_files <= 0 or remaining_total_bytes <= 0:
                    break
                if repo["full_name"] in seen:
                    continue
                seen.add(repo["full_name"])

                skip_reason = should_skip_repo(repo, allow_licenses)
                if skip_reason:
                    rejected_repos.append({"full_name": repo["full_name"], "reason": skip_reason})
                    continue

                repo_dir = temp_root / slugify_repo_name(repo["full_name"])
                try:
                    clone_repository(repo["clone_url"], repo_dir, clone_depth=args.clone_depth)
                    extraction = extract_repository_corpus(
                        repo={
                            "full_name": repo["full_name"],
                            "language": repo.get("language"),
                            "license_spdx": (repo.get("license") or {}).get("spdx_id"),
                            "stargazers_count": repo["stargazers_count"],
                        },
                        repo_dir=repo_dir,
                        repo_output_dir=repos_output_dir,
                        max_files_per_repo=args.max_files_per_repo,
                        min_code_files_per_repo=args.min_code_files_per_repo,
                        max_bytes_per_file=args.max_bytes_per_file,
                        max_bytes_per_repo=args.max_bytes_per_repo,
                        remaining_files=remaining_total_files,
                        remaining_bytes=remaining_total_bytes,
                        allowed_extensions=allowed_extensions,
                    )
                except (subprocess.CalledProcessError, OSError, urllib.error.URLError) as error:
                    rejected_repos.append({"full_name": repo["full_name"], "reason": f"clone-or-read-error:{error}"})
                    shutil.rmtree(repo_dir, ignore_errors=True)
                    continue
                finally:
                    shutil.rmtree(repo_dir, ignore_errors=True)

                if not extraction["accepted"]:
                    rejected_repos.append({"full_name": repo["full_name"], "reason": extraction["reason"]})
                    continue

                remaining_total_files -= len(extraction["files"])
                remaining_total_bytes -= extraction["bytes"]
                language_kept += 1
                selected_repos.append(
                    {
                        "full_name": repo["full_name"],
                        "clone_url": repo["clone_url"],
                        "html_url": repo["html_url"],
                        "description": repo.get("description"),
                        "language": repo.get("language"),
                        "default_branch": repo.get("default_branch"),
                        "stars": repo["stargazers_count"],
                        "license_spdx": (repo.get("license") or {}).get("spdx_id"),
                        "size_kb": repo.get("size"),
                        "repo_corpus_file": extraction["repo_file"],
                        "extracted_files": len(extraction["files"]),
                        "extracted_bytes": extraction["bytes"],
                    }
                )

    dataset_path = output_dir / "train.txt"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as combined:
        for repo in selected_repos:
            combined.write(Path(repo["repo_corpus_file"]).read_text(encoding="utf-8"))

    summary = {
        "languages": languages,
        "repo_limit": args.repo_limit,
        "per_language_limit": args.per_language_limit,
        "min_stars": args.min_stars,
        "strict_language_files": args.strict_language_files,
        "selected_repo_count": len(selected_repos),
        "rejected_repo_count": len(rejected_repos),
        "dataset_path": str(dataset_path),
        "dataset_bytes": dataset_path.stat().st_size if dataset_path.exists() else 0,
        "total_extracted_files": sum(repo["extracted_files"] for repo in selected_repos),
    }

    (output_dir / "selected-repos.json").write_text(
        json.dumps(selected_repos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rejected-repos.json").write_text(
        json.dumps(rejected_repos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not selected_repos or summary["dataset_bytes"] == 0:
        raise SystemExit("No usable repositories were selected for the code corpus.")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

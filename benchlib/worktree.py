from __future__ import annotations

import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pygit2


def _sanitize_worktree_suffix(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return sanitized or "worktree"


def _resolve_label_revision(repo: pygit2.Repository, revision: str) -> str:
    resolved = repo.revparse_single(revision)
    commit_id = getattr(resolved, "id", None)
    if commit_id is None:
        raise ValueError(f"Revision did not resolve to an object id: {revision}")
    return str(commit_id)


def detached_worktree_path(
    repo: pygit2.Repository,
    revision: str,
    prefix: str,
    *,
    label_revision: str | None = None,
) -> Path:
    if repo.workdir is None:
        raise ValueError("Repository does not have a workdir")
    repo_root = Path(repo.workdir).resolve()
    suffix = _sanitize_worktree_suffix(_resolve_label_revision(repo, label_revision or revision))
    return Path.home() / "brokkbench" / repo_root.name / f"{prefix}-{suffix}"


def cleanup_detached_worktree(repo_root: Path, worktree_path: Path) -> None:
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        pass

    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)


def _called_process_message(exc: subprocess.CalledProcessError) -> str:
    parts = [str(exc)]
    stderr = (exc.stderr or "").strip()
    stdout = (exc.stdout or "").strip()
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    return "\n".join(parts)


@contextmanager
def materialize_detached_worktree(
    repo_path: str,
    revision: str,
    *,
    prefix: str = "worktree",
    label_revision: str | None = None,
):
    repo = pygit2.Repository(repo_path)
    if repo.workdir is None:
        raise ValueError(f"Repository at {repo_path} does not have a workdir")

    repo_root = Path(repo.workdir).resolve()
    worktree_path = detached_worktree_path(repo, revision, prefix, label_revision=label_revision)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    cleanup_detached_worktree(repo_root, worktree_path)
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree_path), revision],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        cleanup_detached_worktree(repo_root, worktree_path)
        raise RuntimeError(_called_process_message(exc)) from exc
    try:
        yield worktree_path
    finally:
        cleanup_detached_worktree(repo_root, worktree_path)

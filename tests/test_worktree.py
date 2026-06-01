from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from benchlib.worktree import (
    UnsupportedRepositoryError,
    materialize_detached_worktree,
    repo_requires_git_lfs,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _create_repo(tmp_path: Path, *, gitattributes: str | None = None) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    if gitattributes is not None:
        (repo / ".gitattributes").write_text(gitattributes, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    return repo


def test_repo_requires_git_lfs_detects_gitattributes(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path, gitattributes="*.bin filter=lfs diff=lfs merge=lfs -text\n")

    assert repo_requires_git_lfs(str(repo / ".git")) is True


def test_repo_requires_git_lfs_false_for_normal_repo(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)

    assert repo_requires_git_lfs(str(repo / ".git")) is False


def test_materialize_detached_worktree_rejects_git_lfs_repo_before_git_worktree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _create_repo(tmp_path, gitattributes="*.bin filter=lfs diff=lfs merge=lfs -text\n")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    try:
        with materialize_detached_worktree(str(repo / ".git"), "HEAD"):
            pass
    except UnsupportedRepositoryError as exc:
        assert "Git LFS" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected UnsupportedRepositoryError")

    assert calls == []


def test_materialize_detached_worktree_maps_checkout_git_lfs_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _create_repo(tmp_path)
    real_subprocess = subprocess

    def fake_run(cmd: list[str], *, cwd: Path, check: bool, capture_output: bool, text: bool):
        if cmd[1:4] == ["worktree", "remove", "--force"]:
            return SimpleNamespace(returncode=0)
        raise real_subprocess.CalledProcessError(
            returncode=128,
            cmd=cmd,
            stderr="git-lfs filter-process: 1: git-lfs: not found\nfatal: the remote end hung up unexpectedly\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("benchlib.worktree.repo_requires_git_lfs", lambda _repo_path: False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    try:
        with materialize_detached_worktree(str(repo / ".git"), "HEAD"):
            pass
    except UnsupportedRepositoryError as exc:
        assert "Git LFS" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected UnsupportedRepositoryError")

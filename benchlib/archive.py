import argparse
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile
import re

from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def _get_project_path_from_worktree(worktree_path: pathlib.Path) -> pathlib.Path:
    """
    Derive the main repository directory by reading the linked worktree's .git file.
    """
    git_file = worktree_path / ".git"
    if not git_file.is_file():
        raise ValueError(f".git file not found in worktree: {git_file}")
    with open(git_file, "r", encoding="utf-8") as fp:
        first = fp.readline().strip()
    m = re.match(r"gitdir: (.*?)/\.git/worktrees/[^/]+/?$", first)
    if not m:
        raise ValueError(f"Cannot derive project path from {git_file!s}: {first!r}")
    return pathlib.Path(m.group(1)).resolve()


def _git_generic(root: pathlib.Path, *git_args: str) -> subprocess.CompletedProcess:
    """
    Helper to run git commands and return the completed process.
    Raises CalledProcessError if the command fails.
    """
    try:
        return subprocess.run(
            ["git", *git_args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd} in {root}\n{e.stderr}", file=sys.stderr)
        raise


def cleanup_worktree(project_path: pathlib.Path, worktree_path: pathlib.Path):
    """Force-deletes a git worktree and its associated sidecar logs."""
    if worktree_path.exists():
        logger.info(f"Cleaning up worktree {worktree_path}")
    try:
        _git_generic(project_path, "worktree", "remove", "--force", str(worktree_path))
    except subprocess.CalledProcessError:
        if worktree_path.exists():
            logger.error(f"git worktree remove failed, falling back to rmtree for {worktree_path}")
            shutil.rmtree(worktree_path, ignore_errors=True)

    agent_log_path = worktree_path.parent / f"{worktree_path.name}-agent.txt"
    if agent_log_path.exists():
        try:
            agent_log_path.unlink()
        except OSError as e:
            logger.error(f"Error deleting agent log {agent_log_path}: {e}")
    bootstrap_log_path = worktree_path.parent / f"{worktree_path.name}-bootstrap.txt"
    if bootstrap_log_path.exists():
        try:
            bootstrap_log_path.unlink()
        except OSError as e:
            logger.error(f"Error deleting bootstrap log {bootstrap_log_path}: {e}")
    harness_tests_log_path = worktree_path.parent / f"{worktree_path.name}-harness-tests.txt"
    if harness_tests_log_path.exists():
        try:
            harness_tests_log_path.unlink()
        except OSError as e:
            logger.error(f"Error deleting harness tests log {harness_tests_log_path}: {e}")


def _archive_root() -> pathlib.Path:
    return pathlib.Path.home() / "brokkbench-archive"


def _sanitize_archive_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "archive"


def archive_worktree(
    project_path: pathlib.Path,
    worktree_path: pathlib.Path,
    pre_agent_head: str | None = None,
    archive_name: str | None = None,
) -> pathlib.Path | None:
    """
    Create a zip archive of a worktree's results. Includes artifacts if present:
      - run-output.txt
      - .brokk/llm-history/**
      - .brokk/sft_gen/**
    Always cleans up the worktree afterwards. Returns the created zip Path, or None if the
    worktree does not exist. The `pre_agent_head` argument is retained for call-site compatibility.
    """
    project_path = pathlib.Path(project_path)
    worktree_path = pathlib.Path(worktree_path)

    try:
        if not worktree_path.is_dir():
            return None

        run_output = worktree_path / "run-output.txt"
        llm_history_dir = worktree_path / ".brokk" / "llm-history"
        sft_gen_dir = worktree_path / ".brokk" / "sft_gen"
        bootstrap_log_path = worktree_path.parent / f"{worktree_path.name}-bootstrap.txt"
        harness_tests_log_path = worktree_path.parent / f"{worktree_path.name}-harness-tests.txt"

        archive_dir = _archive_root() / project_path.name
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_stem = worktree_path.name if archive_name is None else _sanitize_archive_name(archive_name)
        if not archive_stem.endswith(".zip"):
            archive_stem = f"{archive_stem}.zip"
        zip_path = archive_dir / archive_stem
        logger.info(f"Archiving worktree {worktree_path} to {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # run-output.txt
            if run_output.exists():
                zf.write(run_output, arcname="run-output.txt")
            else:
                logger.warning(f"{run_output} missing; continuing without it.")

            # .brokk/llm-history/**
            if llm_history_dir.is_dir():
                for root, _, files in os.walk(llm_history_dir):
                    for file in files:
                        file_path = pathlib.Path(root) / file
                        arcname = file_path.relative_to(worktree_path)
                        zf.write(file_path, arcname=str(arcname))
            else:
                logger.warning(f"{llm_history_dir} missing; continuing without it.")

            # .brokk/sft_gen/**
            if sft_gen_dir.is_dir():
                for root, _, files in os.walk(sft_gen_dir):
                    for file in files:
                        file_path = pathlib.Path(root) / file
                        arcname = file_path.relative_to(worktree_path)
                        zf.write(file_path, arcname=str(arcname))

            if bootstrap_log_path.exists():
                zf.write(bootstrap_log_path, arcname="bootstrap.txt")

            if harness_tests_log_path.exists():
                zf.write(harness_tests_log_path, arcname="harness-tests.txt")

        logger.info(f"Successfully created archive {zip_path}")
        return zip_path

    finally:
        cleanup_worktree(project_path, worktree_path)


def _archive_worker(worktree_path_str: str):
    """
    A wrapper for archive_worktree to be used with multiprocessing.
    It handles exceptions to prevent pool crashes and ensures basic cleanup.
    """
    worktree_path = pathlib.Path(worktree_path_str)
    try:
        project_path = _get_project_path_from_worktree(worktree_path)
        archive_worktree(project_path, worktree_path, pre_agent_head=None)
    except (ValueError, subprocess.CalledProcessError) as e:
        logger.error(f"Error processing {worktree_path}: {e}")
        logger.warning(f"Attempting to clean up directory {worktree_path}")
        if worktree_path.is_dir():
            shutil.rmtree(worktree_path, ignore_errors=True)
        agent_log_path = worktree_path.parent / f"{worktree_path.name}-agent.txt"
        if agent_log_path.exists():
            try:
                agent_log_path.unlink()
            except OSError as exc:
                print(f"Error deleting agent log {agent_log_path}: {exc}", file=sys.stderr)


def main():
    """Command-line interface for archiving a Brokk worktree."""
    logging.basicConfig()
    parser = argparse.ArgumentParser(description="Archive one or more Brokk worktrees.")
    parser.add_argument(
        "targets",
        nargs="*",
        help="Path(s) to worktrees to archive. If none are provided, read from stdin.",
    )
    args = parser.parse_args()

    targets = args.targets
    if not targets:
        targets = [line.strip() for line in sys.stdin if line.strip()]

    if not targets:
        parser.print_help(sys.stderr)
        sys.exit(1)

    invalid_targets = [t for t in targets if not pathlib.Path(t).is_dir()]
    if invalid_targets:
        for target in invalid_targets:
            logger.error(f"Error: target path does not exist or is not a directory: {target}")
        sys.exit("Aborting due to invalid targets.")

    with multiprocessing.Pool() as pool:
        list(
            tqdm(
                pool.imap_unordered(_archive_worker, targets),
                total=len(targets),
                desc="Archiving worktrees",
            )
        )


if __name__ == "__main__":
    main()

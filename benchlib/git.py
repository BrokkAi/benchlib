import pathlib
import subprocess
import sys


def git_run(root: pathlib.Path | str, *git_args: str) -> str:
    """Run a git command inside the given directory and return stripped stdout."""
    try:
        result = subprocess.run(
            ["git", *git_args],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
            timeout=300,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(
            f"Error executing command: {exc.cmd} in {root}\n{exc.stderr or exc.output}",
            file=sys.stderr,
        )
        raise

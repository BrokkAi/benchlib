"""
benchlib - A library for running automated code agent benchmarks.

Usage:
    from benchlib import run

    # Use run.run_one_revision, run.run_with_retries, run.run_many_tasks
"""

from . import run

__version__ = "0.1.0"
__all__ = ["run"]

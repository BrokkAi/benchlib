import datetime
import enum
import json
import os
import pathlib
import shutil
import subprocess
import sys
import concurrent.futures
from . import archive
from typing import Callable
import random
import time
from dataclasses import dataclass
from collections import defaultdict


class RunOutcome(enum.Enum):
    SUCCESS = 0
    AGENT_FAILED = 1
    TESTS_FAILED = 2


@dataclass(frozen=True)
class RunResult:
    """
    Public API result for a single run.
    - outcome: SUCCESS | AGENT_FAILED | TESTS_FAILED
    - metrics: parsed metrics dict from agent output (may be None if unavailable)
    - archive: path to created zip, or None if skipped/unavailable
    - patch: text of 02-agent.diff; empty string if no agent commit and no changes
    """
    outcome: RunOutcome
    metrics: dict | None
    archive: pathlib.Path | None
    patch: str


@dataclass(frozen=True)
class TaskKey:
    """
    Public API key for mapping run results.
    Identifies a single (revision, model, run_number, task_id).
    task_id allows multiple tasks per revision (e.g., SWE-bench instances).
    """
    revision: str
    model: str
    run_number: int
    task_id: str


_DEFAULT_CLI_BIN_PATH = pathlib.Path("../brokk/cli")
CLI_BIN = pathlib.Path(os.getenv("BRK_CLI_BIN", str(_DEFAULT_CLI_BIN_PATH)))


def _run_cli(cmd: list[str], log_file: pathlib.Path) -> subprocess.CompletedProcess:
    """
    Helper to execute Brokk CLI commands and append output
    to the supplied log-file. If the BB_DEBUG environment variable
    is set, the output is also echoed to the console.
    """
    full_cmd = cmd

    if os.getenv("BB_DEBUG"):
        print(f"Running command: {' '.join(full_cmd)}", file=sys.stderr)

    with subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc, open(log_file, "ab") as log_fp:
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fp.write(line)
            if os.getenv("BB_DEBUG"):
                try:
                    sys.stderr.buffer.write(line)
                    sys.stderr.flush()
                except AttributeError:
                    sys.stderr.write(line.decode(errors="replace"))
                    sys.stderr.flush()
        proc.wait()

    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=None, stderr=None)


def run_one_revision(
    project: str,
    revision: str,
    results_root: pathlib.Path,
    run_number: int,
    jvm_args: list[str],
    stagger_seconds: int,
    model: str,
    task_id: str,
    get_cli_args: Callable[[str, str], list[str]],
    execute_tests: Callable[[pathlib.Path, pathlib.Path], subprocess.CompletedProcess],
    commit_tests: Callable[[pathlib.Path, pathlib.Path, str], None] | None = None,
) -> RunResult:
    """
    Execute the workflow for a single (revision, model, run_number, task_id).

    - If commit_tests is None:
        * git reset --hard <revision>
      else:
        * git reset --hard <revision>^
        * commit_tests(project_path, worktree_path, revision) should produce a HEAD commit (tests snapshot)
        * write "01-tests.diff" = `git show HEAD`
    - Run agent with CLI args from get_cli_args(revision, task_id).
    - If agent succeeds: commit "Agent work", patch=`git show HEAD`
      If agent fails (no commit): patch=`git diff` (may be empty).
    - Archive with 02-agent.diff and optional 01-tests.diff; return created zip path.
    """
    project_path = pathlib.Path(project).resolve()
    if not (project_path / ".git").is_dir():
        raise ValueError(f"Project '{project}' is not a git repository.")

    # ------------------------------------------------------------------
    # 1. set up the worktree
    # ------------------------------------------------------------------
    def _git_generic(root: pathlib.Path, *git_args: str) -> str:
        """
        Helper to run git commands inside the work-tree and return stdout.
        Raises CalledProcessError if the command fails.
        """
        try:
            return subprocess.run(
                ["git", *git_args],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError as e:
            print(
                f"-----\nError executing command: {e.cmd} in {root}\n{e.output}\n-----\n",
                file=sys.stderr,
            )
            raise

    # Resolve short hash for naming
    revshort = _git_generic(project_path, "rev-parse", "--short", revision)

    session = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    workdir_name = f"{model}-{revshort}-{run_number}-{session}"
    worktree_path = pathlib.Path.home() / "brokkbench" / project_path.name / workdir_name

    # Helper to compute run-output amalgam
    def _write_run_output(paths: list[pathlib.Path]) -> None:
        """
        Combine the supplied log files into `run-output.txt`
        inside the worktree directory. Missing paths are ignored.
        """
        run_output_path = worktree_path / "run-output.txt"
        worktree_path.mkdir(parents=True, exist_ok=True)
        with open(run_output_path, "wb") as out_fp:
            for p in paths:
                if p is None or not p.exists():
                    continue
                out_fp.write(f"-------------- {p}\n".encode())
                out_fp.flush()
                with open(p, "rb") as src_fp:
                    shutil.copyfileobj(src_fp, out_fp)
                out_fp.write(b"\n")

    pre_agent_head: str | None = None
    agent_log_path = worktree_path.parent / f"{workdir_name}-agent.txt"
    tests_log_path = worktree_path / "tests.txt"

    try:
        os.makedirs(worktree_path.parent, exist_ok=True)

        # Create work-tree via Brokk CLI (fixed)
        first_cmd = [
            str(CLI_BIN),
            *jvm_args,
            f"--project={project_path}",
            f"--worktree={worktree_path}",
        ]
        first_ret = _run_cli(first_cmd, agent_log_path)
        if first_ret.returncode != 0:
            _write_run_output([agent_log_path])
            # Archive and return failure
            zip_path = archive.archive_worktree(project_path, worktree_path, pre_agent_head=None)
            return RunResult(RunOutcome.AGENT_FAILED, metrics=None, archive=zip_path, patch="")

        # Reset based on commit_tests presence
        if commit_tests is None:
            _git_generic(worktree_path, "reset", "--hard", revision)
            pre_agent_head = _git_generic(worktree_path, "rev-parse", "HEAD")
        else:
            _git_generic(worktree_path, "reset", "--hard", f"{revision}^")
            head_before = _git_generic(worktree_path, "rev-parse", "HEAD")
            commit_tests(project_path, worktree_path, revision)
            head_after = _git_generic(worktree_path, "rev-parse", "HEAD")
            if head_before == head_after:
                raise ValueError("commit_tests did not create a new HEAD commit as required.")
            # Write 01-tests.diff
            tests_diff = _git_generic(worktree_path, "show", "HEAD")
            with open(worktree_path / "01-tests.diff", "w", encoding="utf-8") as fp:
                fp.write(tests_diff)
            pre_agent_head = head_after

        # Stagger AFTER validations/sanity checks
        if stagger_seconds and stagger_seconds > 0:
            time.sleep(random.uniform(0, stagger_seconds))

        # ------------------------------------------------------------------
        # 2. Run Brokk CLI agent task using bpr-provided args
        # ------------------------------------------------------------------
        agent_args = list(get_cli_args(revision, task_id) or [])
        second_cmd: list[str] = [
            str(CLI_BIN),
            *jvm_args,
            f"--project={project_path}",
            f"--worktree={worktree_path}",
            *agent_args,
        ]
        agent_proc = _run_cli(second_cmd, agent_log_path)

        # Extract metrics from agent log (may be absent on failure)
        metrics_json: str | None = None
        if agent_log_path.exists():
            with open(agent_log_path, "r", encoding="utf-8") as log_fp:
                for line in log_fp:
                    if line.startswith("BRK_CODEAGENT_METRICS="):
                        metrics_json = line.strip().split("=", 1)[1]
                        break
        metrics: dict | None = json.loads(metrics_json) if metrics_json else None
        if metrics is not None:
            metrics["worktree"] = str(worktree_path)

        # ------------------------------------------------------------------
        # 3. Commit agent's work on success; compute patch either way
        # ------------------------------------------------------------------
        patch_text = ""
        if agent_proc.returncode == 0:
            # Commit agent's work
            _git_generic(worktree_path, "add", "-A")
            _git_generic(worktree_path, "commit", "--allow-empty", "-m", "Agent work")
            patch_text = _git_generic(worktree_path, "show", "HEAD")
        else:
            # No commit, compute dirty diff (may be empty)
            try:
                patch_text = _git_generic(worktree_path, "diff")
            except subprocess.CalledProcessError:
                patch_text = ""

        # ------------------------------------------------------------------
        # 4. Write results JSON
        # ------------------------------------------------------------------
        results_dir = pathlib.Path(results_root) / f"{project_path.name}{run_number}"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"{model}-{revision}.json"

        # Execute tests only if agent succeeded (we have a committed state)
        outcome = RunOutcome.AGENT_FAILED
        if agent_proc.returncode == 0:
            try:
                test_cp = execute_tests(project_path, worktree_path)
                tests_failed = (test_cp.returncode != 0)
                if metrics is None:
                    metrics = {}
                if tests_failed:
                    metrics["stopReason"] = "HARNESS_TESTS_FAILED"
                    outcome = RunOutcome.TESTS_FAILED
                else:
                    outcome = RunOutcome.SUCCESS
            except Exception as exc:
                outcome = RunOutcome.TESTS_FAILED
                if metrics is None:
                    metrics = {}
                metrics["stopReason"] = f"HARNESS_EXECUTION_ERROR: {exc}"
            finally:
                # Ensure tests log existence for aggregation
                if not tests_log_path.exists():
                    print("execute_tests did not create tests.txt; logging an error.", file=sys.stderr)

        # Persist metrics JSON (even if minimal) to support retry heuristics
        to_write_metrics = metrics if metrics is not None else {"stopReason": "AGENT_FAILED"}
        with open(results_path, "w", encoding="utf-8") as res_fp:
            json.dump(to_write_metrics, res_fp)
            res_fp.write("\n")

        # Aggregate logs
        log_paths = [agent_log_path]
        if tests_log_path.exists():
            log_paths.append(tests_log_path)
        _write_run_output(log_paths)

        # Archive and cleanup
        zip_path = archive.archive_worktree(project_path, worktree_path, pre_agent_head=pre_agent_head)

        return RunResult(outcome=outcome, metrics=metrics, archive=zip_path, patch=patch_text)

    except Exception:
        # On unexpected exceptions, attempt to archive whatever exists and re-raise
        try:
            archive.archive_worktree(project_path, worktree_path, pre_agent_head=pre_agent_head)
        except Exception:
            pass
        raise


def run_with_retries(
    project: str,
    revision: str,
    results_root: pathlib.Path,
    run_number: int,
    jvm_args: list[str],
    stagger_seconds: int,
    model: str,
    task_id: str,
    get_cli_args: Callable[[str, str], list[str]],
    execute_tests: Callable[[pathlib.Path, pathlib.Path], subprocess.CompletedProcess],
    commit_tests: Callable[[pathlib.Path, pathlib.Path, str], None] | None = None,
) -> tuple[RunResult, bool]:
    """
    Retry `run_one_revision` up to a maximum number of attempts under certain conditions.
    Retries are attempted when:
      - No metrics JSON is produced (e.g., agent crash/OOM-kill), or
      - The metrics JSON's stopExplanation contains any of:
        "too many open files", "check litellm logs", "ratelimiterror".
    """
    MAX_ATTEMPTS = 3
    project_path = pathlib.Path(project).resolve()

    last_result: RunResult | None = None
    hit_retry_max = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = run_one_revision(
            project=project,
            revision=revision,
            results_root=results_root,
            run_number=run_number,
            jvm_args=jvm_args,
            stagger_seconds=stagger_seconds,
            model=model,
            task_id=task_id,
            get_cli_args=get_cli_args,
            execute_tests=execute_tests,
            commit_tests=commit_tests,
        )
        last_result = result

        # Locate the metrics file written by `run_one_revision`.
        results_dir = pathlib.Path(results_root) / f"{project_path.name}{run_number}"
        results_path = results_dir / f"{model}-{revision}.json"

        metrics = None
        if results_path.exists():
            try:
                with open(results_path, "r", encoding="utf-8") as fp:
                    metrics = json.load(fp)
            except Exception:
                metrics = None  # Treat unreadable/broken as missing

        # Decide whether to retry
        should_retry = False
        reason_for_retry = None

        if metrics is None:
            should_retry = True
            reason_for_retry = "no metrics file"
        else:
            stop_expl = str(metrics.get("stopExplanation", "")).lower()
            if (
                "too many open files" in stop_expl
                or "check litellm logs" in stop_expl
                or "ratelimiterror" in stop_expl
            ):
                should_retry = True
                reason_for_retry = stop_expl

        if should_retry and attempt < MAX_ATTEMPTS:
            print(f"Automatically retrying (attempt {attempt + 1}/{MAX_ATTEMPTS}) based on {reason_for_retry}", file=sys.stderr)
            continue
        if should_retry:
            hit_retry_max = True
        break

    assert last_result is not None
    return last_result, hit_retry_max


def run_many_tasks(
    project: str,
    results_root: pathlib.Path,
    threads: int,
    jobs: list[tuple[str, str, int, str]],  # (revision, model, run_number, task_id)
    jvm_args: list[str],
    stagger_seconds: int,
    get_cli_args: Callable[[str, str], list[str]],
    execute_tests: Callable[[pathlib.Path, pathlib.Path], subprocess.CompletedProcess],
    commit_tests: Callable[[pathlib.Path, pathlib.Path, str], None] | None = None,
) -> dict[TaskKey, RunResult]:
    """
    Run multiple jobs concurrently. Each job supplies callables for:
      - get_cli_args(revision, task_id)
      - execute_tests(project_path, worktree_path)
      - commit_tests(project_path, worktree_path, revision) [optional]
    Returns a mapping TaskKey -> RunResult.
    """
    os.environ["BRK_CODEAGENT_METRICS"] = "true"

    results_map: dict[TaskKey, RunResult] = {}
    successful_tasks_by_model: dict[str, set[str]] = defaultdict(set)
    retry_gave_up_by_model: dict[str, set[str]] = defaultdict(set)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_job = {
            executor.submit(
                run_with_retries,
                project,
                rev,
                results_root,
                run_number,
                jvm_args,
                stagger_seconds,
                model,
                task_id,
                get_cli_args,
                execute_tests,
                commit_tests,
            ): (rev, model, run_number, task_id)
            for (rev, model, run_number, task_id) in jobs
        }
        try:
            for future in concurrent.futures.as_completed(future_to_job):
                rev, model, run_number, task_id = future_to_job[future]
                key = TaskKey(revision=rev, model=model, run_number=run_number, task_id=task_id)
                try:
                    res, hit_retry_max = future.result()
                    results_map[key] = res
                    if res.outcome == RunOutcome.SUCCESS:
                        successful_tasks_by_model[model].add(rev)
                    elif hit_retry_max:
                        retry_gave_up_by_model[model].add(rev)
                except Exception as exc:
                    print(
                        f"Fatal error while processing revision '{rev}' with model '{model}' (run {run_number}, task '{task_id}'): {exc}",
                        file=sys.stderr,
                    )
                    raise
        finally:
            executor.shutdown(wait=True)

    if successful_tasks_by_model:
        print("\n--- Successful tasks per model (at least one run) ---", file=sys.stderr)
        model_counts = {
            model: len(revs) for model, revs in successful_tasks_by_model.items()
        }
        sorted_models = sorted(
            model_counts.items(), key=lambda item: item[1], reverse=True
        )
        for model, count in sorted_models:
            print(f"{model}: {count}", file=sys.stderr)

    if any(len(s) > 0 for s in retry_gave_up_by_model.values()):
        print("\n--- Tasks that hit retry max and gave up (per model) ---", file=sys.stderr)
        for model in sorted(retry_gave_up_by_model.keys()):
            revs = sorted(retry_gave_up_by_model[model])
            cdl = ",".join(revs)
            print(f"{model}: {cdl}", file=sys.stderr)

    return results_map

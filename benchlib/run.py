import concurrent.futures
import datetime
import enum
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import traceback
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from . import archive


class RunOutcome(enum.Enum):
    SUCCESS = 0
    AGENT_ERROR = 1
    AGENT_FAILED = 2
    TESTS_FAILED = 3


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
class Task:
    """
    Public API key for mapping run results.

    Identifies a single (project, revision, model, run_number, task_id).
    Additional per-task execution configuration is carried as non-key fields.
    """

    project: str
    revision: str
    model: str
    run_number: int
    task_id: str | None = None
    job_env: dict[str, str] | None = field(default=None, compare=False, hash=False, repr=False)
    heap_mb: int = field(default=0, compare=False, hash=False, repr=False)
    properties: dict[str, str] | None = field(default=None, compare=False, hash=False, repr=False)

    def filename(self) -> str:
        if self.task_id:
            return f"{self.model}-{self.revision}-{self.task_id}.json"
        return f"{self.model}-{self.revision}.json"


def _resolve_cli_bin() -> pathlib.Path:
    env_path = os.getenv("BRK_CLI_BIN")
    if env_path:
        return pathlib.Path(env_path)

    try:
        from . import cli as _cli  # type: ignore
    except Exception:
        _cli = None

    if _cli is not None:
        try:
            candidate = getattr(_cli, "CLI_BIN", None)
            if candidate:
                return pathlib.Path(candidate)
        except Exception:
            pass

        try:
            getter = getattr(_cli, "get_cli_bin", None)
            if callable(getter):
                candidate = getter()
                if candidate:
                    return pathlib.Path(candidate)
        except Exception:
            pass

    return pathlib.Path("../brokk/cli")


def _merged_env(task_env: dict[str, str] | None) -> dict[str, str]:
    env = dict(os.environ)
    env["BRK_COLLECT_METRICS"] = "true"
    if task_env:
        env.update(task_env)
    return env


def _run_cli(cmd: list[str], log_file: pathlib.Path, env: dict[str, str] | None) -> subprocess.CompletedProcess:
    full_cmd = cmd

    if os.getenv("BB_DEBUG"):
        print(f"Running command: {' '.join(full_cmd)}", file=sys.stderr)

    try:
        with subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
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
    except Exception as exc:
        raise RuntimeError(f"Failed to execute command: {' '.join(full_cmd)}") from exc

    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=None, stderr=None)


def _append_task_log(log_path: pathlib.Path, message: str) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(f"[{ts}] {message}\n")
    except Exception:
        # Never fail task execution because of logging problems.
        pass


def _run_one_task(
    task: Task,
    results_root: pathlib.Path,
    jvm_args: list[str],
    stagger_seconds: int,
    get_cli_args: Callable[[Task], list[str]],
    execute_tests: Callable[[pathlib.Path, pathlib.Path, dict[str, str], dict[str, str] | None], subprocess.CompletedProcess]
    | None,
    commit_tests: Callable[[pathlib.Path, pathlib.Path, str, dict[str, str], dict[str, str] | None], None]
    | None = None,
    on_task_start: Callable[[Task, pathlib.Path, int], None] | None = None,
    attempt: int = 1,
) -> RunResult:
    project_path = pathlib.Path(task.project).resolve()
    if not (project_path / ".git").is_dir():
        raise ValueError(f"Project '{task.project}' is not a git repository.")

    env = _merged_env(task.job_env)

    def _git_generic(root: pathlib.Path, *git_args: str) -> str:
        try:
            return subprocess.run(
                ["git", *git_args],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
                env=env,
            ).stdout.strip()
        except subprocess.CalledProcessError as e:
            print(
                f"-----\nError executing command: {e.cmd} in {root}\n{e.output}\n-----\n",
                file=sys.stderr,
            )
            raise

    revshort = _git_generic(project_path, "rev-parse", "--short", task.revision)

    session = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    if task.task_id:
        workdir_name = f"{task.model}-{revshort}-{task.task_id}-{task.run_number}-{session}"
    else:
        workdir_name = f"{task.model}-{revshort}-{task.run_number}-{session}"
    worktree_path = pathlib.Path.home() / "brokkbench" / project_path.name / workdir_name

    def _write_run_output(paths: list[pathlib.Path]) -> None:
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

    cli_bin = _resolve_cli_bin()

    pre_agent_head: str | None = None
    bootstrap_log_path = worktree_path.parent / f"{workdir_name}-bootstrap.txt"
    agent_log_path = worktree_path.parent / f"{workdir_name}-agent.txt"
    tests_log_path = worktree_path / "tests.txt"

    def _log_stage(message: str) -> None:
        _append_task_log(bootstrap_log_path, message)
        if os.getenv("BB_DEBUG"):
            print(f"[{task.model}] {message}", file=sys.stderr)

    try:
        _log_stage(
            f"Starting task: project={project_path} revision={task.revision} run_number={task.run_number} "
            f"task_id={task.task_id} cli={cli_bin}"
        )
        if on_task_start is not None:
            try:
                on_task_start(task, worktree_path, attempt)
            except TypeError:
                on_task_start(task, worktree_path)
        os.makedirs(worktree_path.parent, exist_ok=True)
        _log_stage(f"Created worktree parent: {worktree_path.parent}")

        if stagger_seconds and stagger_seconds > 0:
            _log_stage(f"Applying stagger: up to {stagger_seconds} seconds")
            time.sleep(random.uniform(0, stagger_seconds))

        _log_stage(f"First cli command: {str(cli_bin)} --project={project_path} --worktree={worktree_path}")

        first_cmd = [
            str(cli_bin),
            *jvm_args,
            f"--project={project_path}",
            f"--worktree={worktree_path}",
        ]
        first_ret = _run_cli(first_cmd, agent_log_path, env=env)
        _log_stage(f"First cli return code: {first_ret.returncode}")
        if first_ret.returncode != 0:
            _log_stage(f"First cli failed with return code {first_ret.returncode}")
            _write_run_output([agent_log_path])
            try:
                zip_path = archive.archive_worktree(project_path, worktree_path, pre_agent_head=None)
            except Exception:
                zip_path = None
            return RunResult(RunOutcome.AGENT_FAILED, metrics=None, archive=zip_path, patch="")

        if commit_tests is None:
            _log_stage(f"Checking out revision {task.revision}")
            _git_generic(worktree_path, "reset", "--hard", task.revision)
            pre_agent_head = _git_generic(worktree_path, "rev-parse", "HEAD")
            print(f"Checked out revision {task.revision} ({pre_agent_head[:7]})", file=sys.stderr)
        else:
            _git_generic(worktree_path, "reset", "--hard", f"{task.revision}^")
            head_before = _git_generic(worktree_path, "rev-parse", "HEAD")
            print(f"Checked out revision {task.revision}^ ({head_before[:7]})", file=sys.stderr)
            if commit_tests is None:
                raise RuntimeError("commit_tests callback is required for this execution mode")
            try:
                commit_tests(project_path, worktree_path, task.revision, env, task.properties)
            except TypeError:
                commit_tests(project_path, worktree_path, task.revision, env)
            head_after = _git_generic(worktree_path, "rev-parse", "HEAD")
            if head_before == head_after:
                raise ValueError("commit_tests did not create a new HEAD commit as required.")
            print(f"After commit_tests, HEAD is at {head_after[:7]}", file=sys.stderr)
            tests_diff = _git_generic(worktree_path, "show", "HEAD")
            with open(worktree_path / "01-tests.diff", "w", encoding="utf-8") as fp:
                fp.write(tests_diff)
            pre_agent_head = head_after

        agent_args = list(get_cli_args(task) or [])
        _log_stage(f"Agent args: {agent_args}")
        second_cmd: list[str] = [
            str(cli_bin),
            *jvm_args,
            f"--project={project_path}",
            f"--worktree={worktree_path}",
            *agent_args,
        ]
        _log_stage(f"Second cli command: {' '.join(second_cmd)}")
        agent_proc = _run_cli(second_cmd, agent_log_path, env=env)
        _log_stage(f"Second cli return code: {agent_proc.returncode}")

        code_metrics_json: str | None = None
        search_metrics_json: str | None = None
        if agent_log_path.exists():
            with open(agent_log_path, "r", encoding="utf-8") as log_fp:
                for line in log_fp:
                    if line.startswith("BRK_CODEAGENT_METRICS="):
                        code_metrics_json = line.strip().split("=", 1)[1]
                    elif line.startswith("BRK_SEARCHAGENT_METRICS="):
                        search_metrics_json = line.strip().split("=", 1)[1]

        metrics_json = code_metrics_json or search_metrics_json
        metrics: dict = json.loads(metrics_json) if metrics_json else {"stopReason": "UNKNOWN"}
        metrics["worktree"] = str(worktree_path)
        metrics["exit_code"] = agent_proc.returncode

        patch_text = ""
        if agent_proc.returncode == 0:
            _git_generic(worktree_path, "add", "-A")
            _git_generic(worktree_path, "commit", "--allow-empty", "-m", "Agent work")
            patch_text = _git_generic(worktree_path, "show", "HEAD")
        else:
            try:
                patch_text = _git_generic(worktree_path, "diff")
            except subprocess.CalledProcessError:
                patch_text = ""

        results_dir = pathlib.Path(results_root) / f"{project_path.name}{task.run_number}"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / task.filename()

        if agent_proc.returncode != 0:
            outcome = RunOutcome.AGENT_ERROR
            metrics["stopReason"] = "AGENT_ERROR"
        elif metrics.get("stopReason") == "SUCCESS":
            if execute_tests is None:
                outcome = RunOutcome.SUCCESS
            else:
                try:
                    try:
                        test_cp = execute_tests(project_path, worktree_path, env, task.properties)
                    except TypeError:
                        test_cp = execute_tests(project_path, worktree_path, env)
                    if test_cp.returncode != 0:
                        metrics["stopReason"] = "HARNESS_TESTS_FAILED"
                        outcome = RunOutcome.TESTS_FAILED
                    else:
                        outcome = RunOutcome.SUCCESS
                except Exception as exc:
                    outcome = RunOutcome.TESTS_FAILED
                    metrics["stopReason"] = f"HARNESS_EXECUTION_ERROR: {exc}"
                finally:
                    if not tests_log_path.exists():
                        print("execute_tests did not create tests.txt; logging an error.", file=sys.stderr)
        else:
            outcome = RunOutcome.AGENT_FAILED

        with open(results_path, "w", encoding="utf-8") as res_fp:
            json.dump(metrics, res_fp, indent=2)
            res_fp.write("\n")

        log_paths = [agent_log_path]
        if tests_log_path.exists():
            log_paths.append(tests_log_path)
        _write_run_output(log_paths)

        try:
            zip_path = archive.archive_worktree(project_path, worktree_path, pre_agent_head=pre_agent_head)
        except Exception:
            zip_path = None
        _log_stage(f"Archived worktree to {zip_path}")

        return RunResult(outcome=outcome, metrics=metrics, archive=zip_path, patch=patch_text)

    except Exception:
        _append_task_log(bootstrap_log_path, "Unhandled exception in task execution")
        _append_task_log(bootstrap_log_path, traceback.format_exc())
        try:
            archive.archive_worktree(project_path, worktree_path, pre_agent_head=pre_agent_head)
        except Exception:
            pass
        raise


def _run_with_retries(
    task: Task,
    results_root: pathlib.Path,
    jvm_args: list[str],
    stagger_seconds: int,
    get_cli_args: Callable[[Task], list[str]],
    execute_tests: Callable[[pathlib.Path, pathlib.Path, dict[str, str], dict[str, str] | None], subprocess.CompletedProcess]
    | None,
    commit_tests: Callable[[pathlib.Path, pathlib.Path, str, dict[str, str], dict[str, str] | None], None]
    | None = None,
    on_task_start: Callable[[Task, pathlib.Path, int], None] | None = None,
) -> tuple[RunResult, bool]:
    MAX_ATTEMPTS = 3
    project_path = pathlib.Path(task.project).resolve()

    last_result: RunResult | None = None
    hit_retry_max = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = _run_one_task(
            task=task,
            results_root=results_root,
            jvm_args=jvm_args,
            stagger_seconds=stagger_seconds,
            get_cli_args=get_cli_args,
            execute_tests=execute_tests,
            commit_tests=commit_tests,
            on_task_start=on_task_start,
            attempt=attempt,
        )
        last_result = result

        results_dir = pathlib.Path(results_root) / f"{project_path.name}{task.run_number}"
        results_path = results_dir / task.filename()

        metrics = None
        if results_path.exists():
            try:
                with open(results_path, "r", encoding="utf-8") as fp:
                    metrics = json.load(fp)
            except Exception:
                metrics = None

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
            print(
                f"Automatically retrying (attempt {attempt + 1}/{MAX_ATTEMPTS}) based on {reason_for_retry}",
                file=sys.stderr,
            )
            continue
        if should_retry:
            hit_retry_max = True
        break

    assert last_result is not None
    return last_result, hit_retry_max


def run_many_tasks(
    *,
    tasks: list[Task],
    results_root: pathlib.Path,
    threads: int,
    jvm_args: list[str],
    stagger_seconds: int,
    get_cli_args: Callable[[Task], list[str]],
    execute_tests: Callable[[pathlib.Path, pathlib.Path, dict[str, str], dict[str, str] | None], subprocess.CompletedProcess]
    | None,
    commit_tests: Callable[[pathlib.Path, pathlib.Path, str, dict[str, str], dict[str, str] | None], None]
    | None = None,
    on_task_start: Callable[[Task, pathlib.Path, int], None] | None = None,
    max_heap_mb: int | None = None,
) -> dict[Task, RunResult]:
    """
    Run tasks concurrently using a thread pool.

    Parameters match the front-end runner expectations:
      - tasks: list[Task] (each Task includes its project path and per-task env)
      - threads: max concurrent processes
      - max_heap_mb: optional global heap budget across concurrent tasks, enforced using task.heap_mb
      - get_cli_args(task): provides CLI args for the agent invocation
      - execute_tests(project_path, worktree_path, env): optional; pass None to skip tests
      - commit_tests(project_path, worktree_path, revision, env): optional
    """
    if threads < 1:
        raise ValueError("threads must be >= 1")

    if not tasks:
        return {}

    os.environ["BRK_COLLECT_METRICS"] = "true"

    results_map: dict[Task, RunResult] = {}
    successful_tasks_by_model: dict[str, set[str]] = defaultdict(set)
    retry_gave_up_by_model: dict[str, set[str]] = defaultdict(set)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_task = {
            executor.submit(
                _run_with_retries,
                task,
                results_root,
                jvm_args,
                stagger_seconds,
                get_cli_args,
                execute_tests,
                commit_tests,
                on_task_start,
            ): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            res, hit_retry_max = future.result()
            results_map[task] = res
            if res.outcome == RunOutcome.SUCCESS:
                successful_tasks_by_model[task.model].add(task.revision)
            elif hit_retry_max:
                retry_gave_up_by_model[task.model].add(task.revision)

    if successful_tasks_by_model:
        print("\n--- Successful tasks per model (at least one run) ---", file=sys.stderr)
        model_counts = {model: len(revs) for model, revs in successful_tasks_by_model.items()}
        sorted_models = sorted(model_counts.items(), key=lambda item: item[1], reverse=True)
        for model, count in sorted_models:
            print(f"{model}: {count}", file=sys.stderr)

    if any(len(s) > 0 for s in retry_gave_up_by_model.values()):
        print("\n--- Tasks that hit retry max and gave up (per model) ---", file=sys.stderr)
        for model in sorted(retry_gave_up_by_model.keys()):
            revs = sorted(retry_gave_up_by_model[model])
            cdl = ",".join(revs)
            print(f"{model}: {cdl}", file=sys.stderr)

    return results_map

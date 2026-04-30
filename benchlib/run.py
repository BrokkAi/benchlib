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
import threading
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
    jvm_args: list[str] | None = field(default=None, compare=False, hash=False, repr=False)
    properties: dict[str, str] | None = field(default=None, compare=False, hash=False, repr=False)

    def filename(self) -> str:
        if self.task_id:
            return f"{self.model}-{self.revision}-{self.task_id}.json"
        return f"{self.model}-{self.revision}.json"


class HeapBudget:
    def __init__(self, max_mb: int):
        self._max = max_mb
        self._used = 0
        self._cond = threading.Condition()

    def acquire(self, mb: int) -> None:
        weight = max(0, int(mb))
        with self._cond:
            while self._used + weight > self._max:
                self._cond.wait()
            self._used += weight

    def release(self, mb: int) -> None:
        weight = max(0, int(mb))
        with self._cond:
            self._used -= weight
            if self._used < 0:
                self._used = 0
            self._cond.notify_all()


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


def _parse_agent_metrics(agent_log_path: pathlib.Path, worktree_path: pathlib.Path) -> dict | None:
    metrics_line = None
    metrics_key = None
    accepted_prefixes = ("BRK_CODEAGENT_METRICS=", "BRK_SEARCHAGENT_METRICS=")

    with open(agent_log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            for prefix in accepted_prefixes:
                if stripped.startswith(prefix):
                    metrics_line = stripped
                    metrics_key = prefix[:-1]
                    break

    if metrics_line is None:
        return None

    payload = metrics_line.split("=", 1)[1]
    try:
        metrics = json.loads(payload)
    except Exception:
        return None

    if not isinstance(metrics, dict):
        return None

    metrics["worktree"] = str(worktree_path)
    metrics.setdefault("metricsType", metrics_key)
    if "stopReason" not in metrics and "stop_reason" in metrics:
        metrics["stopReason"] = metrics["stop_reason"]
    if "stopExplanation" not in metrics and "failure_type" in metrics:
        failure_type = metrics.get("failure_type")
        if isinstance(failure_type, str):
            metrics["stopExplanation"] = failure_type
    return metrics


def _stop_reason(metrics: dict | None) -> str | None:
    if not isinstance(metrics, dict):
        return None
    stop_reason = metrics.get("stopReason")
    return stop_reason if isinstance(stop_reason, str) else None


def _normalize_metrics_stop_reason(metrics: dict | None) -> dict | None:
    if not isinstance(metrics, dict):
        return metrics
    changed_files = metrics.get("changedFiles")
    if metrics.get("stopReason") == "SUCCESS" and isinstance(changed_files, list) and len(changed_files) == 0:
        metrics = dict(metrics)
        metrics["stopReason"] = "NO_EDITS"
        metrics.setdefault("stopExplanation", "Agent reported success without changing any files")
    return metrics


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
    tests_log_path = worktree_path.parent / f"{workdir_name}-harness-tests.txt"

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

            _log_stage("Running commit_tests callback")
            try:
                if task.properties is not None:
                    commit_tests(project_path, worktree_path, task.revision, env, task.properties)
                else:
                    commit_tests(project_path, worktree_path, task.revision, env)
            except TypeError:
                commit_tests(project_path, worktree_path, task.revision, env)
            pre_agent_head = _git_generic(worktree_path, "rev-parse", "HEAD")
            print(f"Commit tests complete, HEAD at {pre_agent_head[:7]}", file=sys.stderr)

        cli_args = get_cli_args(task)
        _log_stage(f"Second cli args: {' '.join(cli_args)}")
        second_cmd = [
            str(cli_bin),
            *jvm_args,
            f"--project={project_path}",
            f"--worktree={worktree_path}",
            *cli_args,
        ]
        second_ret = _run_cli(second_cmd, agent_log_path, env=env)
        _log_stage(f"Second cli return code: {second_ret.returncode}")
        if second_ret.returncode != 0:
            _log_stage(f"Second cli failed with return code {second_ret.returncode}")
            _write_run_output([agent_log_path])
            try:
                zip_path = archive.archive_worktree(project_path, worktree_path, pre_agent_head=pre_agent_head)
            except Exception:
                zip_path = None
            return RunResult(RunOutcome.AGENT_FAILED, metrics=None, archive=zip_path, patch="")

        metrics = _parse_agent_metrics(agent_log_path, worktree_path)
        if metrics is None:
            _log_stage("No supported metrics line found in agent log")

        metrics = _normalize_metrics_stop_reason(metrics)
        stop_reason = _stop_reason(metrics)
        tests_failed = False
        if execute_tests is not None and stop_reason == "SUCCESS":
            _log_stage("Running execute_tests callback")
            try:
                try:
                    if task.properties is not None:
                        test_cp = execute_tests(project_path, worktree_path, env, task.properties)
                    else:
                        test_cp = execute_tests(project_path, worktree_path, env)
                except TypeError:
                    test_cp = execute_tests(project_path, worktree_path, env)
                with open(tests_log_path, "wb") as tf:
                    tf.write((test_cp.stdout or "").encode(errors="replace"))
                    tf.write((test_cp.stderr or "").encode(errors="replace"))
                tests_failed = test_cp.returncode != 0
            except Exception:
                tests_failed = True

        patch_text = ""
        try:
            if pre_agent_head:
                patch_text = _git_generic(worktree_path, "diff", f"{pre_agent_head}..HEAD")
        except Exception:
            patch_text = ""

        results_dir = pathlib.Path(results_root) / f"{project_path.name}{task.run_number}"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / task.filename()

        outcome = RunOutcome.SUCCESS
        if metrics is None:
            outcome = RunOutcome.AGENT_FAILED
        elif stop_reason != "SUCCESS":
            outcome = RunOutcome.AGENT_FAILED
        elif tests_failed:
            outcome = RunOutcome.TESTS_FAILED

        _write_run_output([agent_log_path, tests_log_path])
        try:
            zip_path = archive.archive_worktree(project_path, worktree_path, pre_agent_head=pre_agent_head)
        except Exception:
            zip_path = None

        if isinstance(metrics, dict):
            persisted_metrics = dict(metrics)
            if zip_path is not None:
                persisted_metrics["worktree"] = str(zip_path)
            try:
                with open(results_path, "w", encoding="utf-8") as fp:
                    json.dump(persisted_metrics, fp, indent=2)
                    fp.write("\n")
            except Exception:
                _log_stage(f"Failed to write metrics file: {results_path}")

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

    if max_heap_mb is not None and max_heap_mb < 1:
        raise ValueError("max_heap_mb must be >= 1")

    if max_heap_mb is not None:
        oversized = [task for task in tasks if task.heap_mb > 0 and task.heap_mb > max_heap_mb]
        if oversized:
            first = oversized[0]
            raise ValueError(
                f"Task heap_mb={first.heap_mb} exceeds max_heap_mb={max_heap_mb} "
                f"for {first.project}:{first.revision}/{first.model}/{first.run_number}"
            )

    heap_budget = HeapBudget(max_heap_mb) if max_heap_mb is not None else None

    results_map: dict[Task, RunResult] = {}
    successful_tasks_by_model: dict[str, set[str]] = defaultdict(set)
    retry_gave_up_by_model: dict[str, set[str]] = defaultdict(set)

    def _effective_jvm_args(task: Task) -> list[str]:
        if task.jvm_args is None:
            return jvm_args
        return list(task.jvm_args)

    def _heap_aware_wrapper(task: Task) -> tuple[RunResult, bool]:
        assert heap_budget is not None
        weight = max(0, int(task.heap_mb))
        heap_budget.acquire(weight)
        try:
            return _run_with_retries(
                task,
                results_root,
                _effective_jvm_args(task),
                stagger_seconds,
                get_cli_args,
                execute_tests,
                commit_tests,
                on_task_start,
            )
        finally:
            heap_budget.release(weight)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_task: dict[concurrent.futures.Future, Task] = {}
        for task in tasks:
            if heap_budget is None:
                future = executor.submit(
                    _run_with_retries,
                    task,
                    results_root,
                    _effective_jvm_args(task),
                    stagger_seconds,
                    get_cli_args,
                    execute_tests,
                    commit_tests,
                    on_task_start,
                )
            else:
                future = executor.submit(_heap_aware_wrapper, task)
            future_to_task[future] = task

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

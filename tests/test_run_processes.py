import json
import os
import pathlib
import subprocess
import sys
import zipfile

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import benchlib.run  # noqa: E402
import benchlib.archive  # noqa: E402


def _git(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def test_run_many_tasks_process_mode_smoke(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "repo"
    project.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(["git", "init"], cwd=project)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=project)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=project)

    (project / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "README.md"], cwd=project)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=project)

    rev = _git(project, "rev-parse", "HEAD")

    fake_cli = tmp_path / "fake_cli.py"
    fake_cli.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import os",
                "import pathlib",
                "import subprocess",
                "import sys",
                "",
                "def main() -> int:",
                "    project = None",
                "    worktree = None",
                "    for a in sys.argv[1:]:",
                "        if a.startswith('--project='):",
                "            project = a.split('=', 1)[1]",
                "        elif a.startswith('--worktree='):",
                "            worktree = a.split('=', 1)[1]",
                "    if project and worktree:",
                "        wt = pathlib.Path(worktree)",
                "        if not wt.exists():",
                "            wt.parent.mkdir(parents=True, exist_ok=True)",
                "            subprocess.check_call(['git', '-C', project, 'worktree', 'add', '--detach', worktree])",
                "    sys.stdout.write('BRK_CODEAGENT_METRICS={\"stopReason\":\"SUCCESS\",\"elapsedMillis\":1,\"llmMillis\":1}\\n')",
                "    return 0",
                "",
                "if __name__ == '__main__':",
                "    raise SystemExit(main())",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    os.environ["BRK_CLI_BIN"] = str(fake_cli)

    def get_cli_args(_task: benchlib.run.Task) -> list[str]:
        return []

    def execute_tests(_project_path: pathlib.Path, worktree_path: pathlib.Path, env: dict[str, str]) -> subprocess.CompletedProcess:
        log_path = worktree_path / "tests.txt"
        log_path.write_text("ok\n", encoding="utf-8")
        return subprocess.run(["/bin/true"], env=env, cwd=worktree_path)

    results_root = tmp_path / "results"

    t = benchlib.run.Task(
        project=str(project),
        revision=rev,
        model="dummy",
        run_number=1,
        job_env={"FOO": "BAR"},
        heap_mb=1,
    )

    results = benchlib.run.run_many_tasks(
        tasks=[t],
        results_root=results_root,
        threads=1,
        jvm_args=[],
        stagger_seconds=0,
        get_cli_args=get_cli_args,
        execute_tests=execute_tests,
        commit_tests=None,
        max_heap_mb=8,
    )

    assert t in results
    assert results[t].outcome == benchlib.run.RunOutcome.SUCCESS

    result_file = results_root / f"{project.name}1" / f"{t.model}-{t.revision}.json"
    assert result_file.exists()
    data = json.loads(result_file.read_text(encoding="utf-8"))
    assert data["stopReason"] == "SUCCESS"
    assert data["worktree"].endswith(".zip")
    assert "brokkbench-archive" in data["worktree"]
    assert results[t].archive is not None
    assert data["worktree"] == str(results[t].archive)
    with zipfile.ZipFile(results[t].archive) as zf:
        assert "bootstrap.txt" in zf.namelist()
        assert "harness-tests.txt" in zf.namelist()


def test_run_many_tasks_accepts_searchagent_metrics(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "repo"
    project.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(["git", "init"], cwd=project)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=project)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=project)

    (project / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "README.md"], cwd=project)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=project)

    rev = _git(project, "rev-parse", "HEAD")

    fake_cli = tmp_path / "fake_search_cli.py"
    fake_cli.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import pathlib",
                "import subprocess",
                "import sys",
                "",
                "def main() -> int:",
                "    project = None",
                "    worktree = None",
                "    for a in sys.argv[1:]:",
                "        if a.startswith('--project='):",
                "            project = a.split('=', 1)[1]",
                "        elif a.startswith('--worktree='):",
                "            worktree = a.split('=', 1)[1]",
                "    if project and worktree:",
                "        wt = pathlib.Path(worktree)",
                "        if not wt.exists():",
                "            wt.parent.mkdir(parents=True, exist_ok=True)",
                "            subprocess.check_call(['git', '-C', project, 'worktree', 'add', '--detach', worktree])",
                "    sys.stdout.write('BRK_SEARCHAGENT_METRICS={\"stop_reason\":\"SUCCESS\",\"elapsed_ms\":1,\"success\":true}\\n')",
                "    return 0",
                "",
                "if __name__ == '__main__':",
                "    raise SystemExit(main())",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    old_cli_bin = os.environ.get("BRK_CLI_BIN")
    os.environ["BRK_CLI_BIN"] = str(fake_cli)
    try:
        def get_cli_args(_task: benchlib.run.Task) -> list[str]:
            return ["--search-workspace=test query"]

        results_root = tmp_path / "results"

        t = benchlib.run.Task(
            project=str(project),
            revision=rev,
            model="dummy",
            run_number=1,
            heap_mb=1,
        )

        results = benchlib.run.run_many_tasks(
            tasks=[t],
            results_root=results_root,
            threads=1,
            jvm_args=[],
            stagger_seconds=0,
            get_cli_args=get_cli_args,
            execute_tests=None,
            commit_tests=None,
            max_heap_mb=8,
        )
    finally:
        if old_cli_bin is None:
            os.environ.pop("BRK_CLI_BIN", None)
        else:
            os.environ["BRK_CLI_BIN"] = old_cli_bin

    assert t in results
    assert results[t].outcome == benchlib.run.RunOutcome.SUCCESS
    assert results[t].metrics is not None
    assert results[t].metrics["metricsType"] == "BRK_SEARCHAGENT_METRICS"
    assert results[t].metrics["stopReason"] == "SUCCESS"

    result_file = results_root / f"{project.name}1" / f"{t.model}-{t.revision}.json"
    assert result_file.exists()
    data = result_file.read_text(encoding="utf-8")
    assert "BRK_SEARCHAGENT_METRICS" in data


def test_archive_worktree_writes_zip_to_archive_root(tmp_path: pathlib.Path, monkeypatch) -> None:
    project = tmp_path / "repo"
    project.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(["git", "init"], cwd=project)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=project)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=project)

    (project / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "README.md"], cwd=project)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=project)

    worktree = tmp_path / "brokkbench" / project.name / "wt-1"
    worktree.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "-C", str(project), "worktree", "add", "--detach", str(worktree)])

    (worktree / "run-output.txt").write_text("output\n", encoding="utf-8")
    llm_history = worktree / ".brokk" / "llm-history"
    llm_history.mkdir(parents=True, exist_ok=True)
    (llm_history / "session.log").write_text("history\n", encoding="utf-8")
    bootstrap_log = worktree.parent / f"{worktree.name}-bootstrap.txt"
    bootstrap_log.write_text("bootstrap\n", encoding="utf-8")
    harness_tests_log = worktree.parent / f"{worktree.name}-harness-tests.txt"
    harness_tests_log.write_text("tests\n", encoding="utf-8")

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: fake_home)

    zip_path = benchlib.archive.archive_worktree(project, worktree, pre_agent_head=None)

    expected_path = fake_home / "brokkbench-archive" / project.name / f"{worktree.name}.zip"
    assert zip_path == expected_path
    assert expected_path.exists()
    assert not worktree.exists()
    assert not bootstrap_log.exists()
    assert not harness_tests_log.exists()
    with zipfile.ZipFile(expected_path) as zf:
        assert "bootstrap.txt" in zf.namelist()
        assert "harness-tests.txt" in zf.namelist()


def test_run_many_tasks_preserves_live_worktree_in_json_when_archive_fails(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    project = tmp_path / "repo"
    project.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(["git", "init"], cwd=project)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=project)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=project)

    (project / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "README.md"], cwd=project)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=project)

    rev = _git(project, "rev-parse", "HEAD")

    fake_cli = tmp_path / "fake_cli.py"
    fake_cli.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import pathlib",
                "import subprocess",
                "import sys",
                "",
                "project = None",
                "worktree = None",
                "for a in sys.argv[1:]:",
                "    if a.startswith('--project='):",
                "        project = a.split('=', 1)[1]",
                "    elif a.startswith('--worktree='):",
                "        worktree = a.split('=', 1)[1]",
                "if project and worktree:",
                "    wt = pathlib.Path(worktree)",
                "    if not (wt / '.git').exists():",
                "        wt.parent.mkdir(parents=True, exist_ok=True)",
                "        subprocess.check_call(['git', '-C', project, 'worktree', 'add', '--detach', worktree])",
                "sys.stdout.write('BRK_CODEAGENT_METRICS={\"stopReason\":\"SUCCESS\",\"elapsedMillis\":1,\"llmMillis\":1}\\n')",
                "sys.exit(0)",
            ]
        ),
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    old_cli_bin = os.environ.get("BRK_CLI_BIN")
    os.environ["BRK_CLI_BIN"] = str(fake_cli)

    def get_cli_args(_task: benchlib.run.Task) -> list[str]:
        return []

    def execute_tests(
        _project_path: pathlib.Path,
        worktree_path: pathlib.Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess:
        log_path = worktree_path / "tests.txt"
        log_path.write_text("ok\n", encoding="utf-8")
        return subprocess.run(["/bin/true"], env=env, cwd=worktree_path)

    def raising_archive(
        _project_path: pathlib.Path,
        _worktree_path: pathlib.Path,
        pre_agent_head: str | None = None,
    ) -> pathlib.Path | None:
        raise RuntimeError("archive failed")

    monkeypatch.setattr(benchlib.archive, "archive_worktree", raising_archive)

    results_root = tmp_path / "results"
    t = benchlib.run.Task(project=str(project), revision=rev, model="fake", run_number=1)
    seen_worktrees: list[pathlib.Path] = []

    def on_task_start(
        _task: benchlib.run.Task,
        worktree_path: pathlib.Path,
        _attempt: int,
    ) -> None:
        seen_worktrees.append(worktree_path)

    try:
        results = benchlib.run.run_many_tasks(
            tasks=[t],
            results_root=results_root,
            threads=1,
            jvm_args=[],
            stagger_seconds=0,
            get_cli_args=get_cli_args,
            execute_tests=execute_tests,
            commit_tests=None,
            max_heap_mb=8,
            on_task_start=on_task_start,
        )
    finally:
        if old_cli_bin is None:
            os.environ.pop("BRK_CLI_BIN", None)
        else:
            os.environ["BRK_CLI_BIN"] = old_cli_bin

    assert len(seen_worktrees) == 1
    assert t in results
    assert results[t].outcome == benchlib.run.RunOutcome.SUCCESS
    assert results[t].archive is None

    result_file = results_root / f"{project.name}1" / f"{t.model}-{t.revision}.json"
    assert result_file.exists()
    data = json.loads(result_file.read_text(encoding="utf-8"))
    assert data["stopReason"] == "SUCCESS"
    assert data["worktree"] == str(seen_worktrees[0])


def test_run_many_tasks_treats_no_edits_as_agent_failed_and_skips_tests(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "repo"
    project.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(["git", "init"], cwd=project)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=project)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=project)

    (project / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "README.md"], cwd=project)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=project)

    rev = _git(project, "rev-parse", "HEAD")

    fake_cli = tmp_path / "fake_cli_no_edits.py"
    fake_cli.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import pathlib",
                "import subprocess",
                "import sys",
                "",
                "def main() -> int:",
                "    project = None",
                "    worktree = None",
                "    for a in sys.argv[1:]:",
                "        if a.startswith('--project='):",
                "            project = a.split('=', 1)[1]",
                "        elif a.startswith('--worktree='):",
                "            worktree = a.split('=', 1)[1]",
                "    if project and worktree:",
                "        wt = pathlib.Path(worktree)",
                "        if not wt.exists():",
                "            wt.parent.mkdir(parents=True, exist_ok=True)",
                "            subprocess.check_call(['git', '-C', project, 'worktree', 'add', '--detach', worktree])",
                "    sys.stdout.write('BRK_CODEAGENT_METRICS={\"stopReason\":\"NO_EDITS\",\"elapsedMillis\":1,\"llmMillis\":1}\\n')",
                "    return 0",
                "",
                "if __name__ == '__main__':",
                "    raise SystemExit(main())",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    old_cli_bin = os.environ.get("BRK_CLI_BIN")
    os.environ["BRK_CLI_BIN"] = str(fake_cli)
    execute_tests_called = False

    def get_cli_args(_task: benchlib.run.Task) -> list[str]:
        return []

    def execute_tests(
        _project_path: pathlib.Path,
        worktree_path: pathlib.Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess:
        nonlocal execute_tests_called
        execute_tests_called = True
        log_path = worktree_path / "tests.txt"
        log_path.write_text("should not run\n", encoding="utf-8")
        return subprocess.run(["/bin/true"], env=env, cwd=worktree_path)

    results_root = tmp_path / "results"

    t = benchlib.run.Task(
        project=str(project),
        revision=rev,
        model="dummy",
        run_number=1,
        heap_mb=1,
    )

    try:
        results = benchlib.run.run_many_tasks(
            tasks=[t],
            results_root=results_root,
            threads=1,
            jvm_args=[],
            stagger_seconds=0,
            get_cli_args=get_cli_args,
            execute_tests=execute_tests,
            commit_tests=None,
            max_heap_mb=8,
        )
    finally:
        if old_cli_bin is None:
            os.environ.pop("BRK_CLI_BIN", None)
        else:
            os.environ["BRK_CLI_BIN"] = old_cli_bin

    assert execute_tests_called is False
    assert t in results
    assert results[t].outcome == benchlib.run.RunOutcome.AGENT_FAILED

    result_file = results_root / f"{project.name}1" / f"{t.model}-{t.revision}.json"
    data = json.loads(result_file.read_text(encoding="utf-8"))
    assert data["stopReason"] == "NO_EDITS"


def test_run_many_tasks_normalizes_success_with_empty_changed_files_to_no_edits(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "repo"
    project.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(["git", "init"], cwd=project)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=project)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=project)

    (project / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "README.md"], cwd=project)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=project)

    rev = _git(project, "rev-parse", "HEAD")

    fake_cli = tmp_path / "fake_cli_success_empty_changed_files.py"
    fake_cli.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import pathlib",
                "import subprocess",
                "import sys",
                "",
                "def main() -> int:",
                "    project = None",
                "    worktree = None",
                "    for a in sys.argv[1:]:",
                "        if a.startswith('--project='):",
                "            project = a.split('=', 1)[1]",
                "        elif a.startswith('--worktree='):",
                "            worktree = a.split('=', 1)[1]",
                "    if project and worktree:",
                "        wt = pathlib.Path(worktree)",
                "        if not wt.exists():",
                "            wt.parent.mkdir(parents=True, exist_ok=True)",
                "            subprocess.check_call(['git', '-C', project, 'worktree', 'add', '--detach', worktree])",
                "    sys.stdout.write('BRK_CODEAGENT_METRICS={\"stopReason\":\"SUCCESS\",\"changedFiles\":[],\"elapsedMillis\":1,\"llmMillis\":1}\\n')",
                "    return 0",
                "",
                "if __name__ == '__main__':",
                "    raise SystemExit(main())",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    old_cli_bin = os.environ.get("BRK_CLI_BIN")
    os.environ["BRK_CLI_BIN"] = str(fake_cli)
    execute_tests_called = False

    def get_cli_args(_task: benchlib.run.Task) -> list[str]:
        return []

    def execute_tests(
        _project_path: pathlib.Path,
        worktree_path: pathlib.Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess:
        nonlocal execute_tests_called
        execute_tests_called = True
        log_path = worktree_path / "tests.txt"
        log_path.write_text("should not run\n", encoding="utf-8")
        return subprocess.run(["/bin/true"], env=env, cwd=worktree_path)

    results_root = tmp_path / "results"

    t = benchlib.run.Task(
        project=str(project),
        revision=rev,
        model="dummy",
        run_number=1,
        heap_mb=1,
    )

    try:
        results = benchlib.run.run_many_tasks(
            tasks=[t],
            results_root=results_root,
            threads=1,
            jvm_args=[],
            stagger_seconds=0,
            get_cli_args=get_cli_args,
            execute_tests=execute_tests,
            commit_tests=None,
            max_heap_mb=8,
        )
    finally:
        if old_cli_bin is None:
            os.environ.pop("BRK_CLI_BIN", None)
        else:
            os.environ["BRK_CLI_BIN"] = old_cli_bin

    assert execute_tests_called is False
    assert t in results
    assert results[t].outcome == benchlib.run.RunOutcome.AGENT_FAILED
    assert results[t].metrics is not None
    assert results[t].metrics["stopReason"] == "NO_EDITS"

    result_file = results_root / f"{project.name}1" / f"{t.model}-{t.revision}.json"
    data = json.loads(result_file.read_text(encoding="utf-8"))
    assert data["stopReason"] == "NO_EDITS"
    assert data["changedFiles"] == []

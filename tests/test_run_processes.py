import os
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import benchlib.run


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
    data = result_file.read_text(encoding="utf-8")
    assert "SUCCESS" in data
    assert "worktree" in data


def test_archive_worktree_writes_zip_to_archive_root(tmp_path: pathlib.Path, monkeypatch) -> None:
    import benchlib.archive  # noqa: E402

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

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: fake_home)

    zip_path = benchlib.archive.archive_worktree(project, worktree, pre_agent_head=None)

    expected_path = fake_home / "brokkbench-archive" / project.name / f"{worktree.name}.zip"

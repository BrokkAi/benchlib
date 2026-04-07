from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchlib.test_commands import build_testsome_command, render_test_command  # noqa: E402


def test_render_test_command_supports_files_and_fqclasses() -> None:
    assert render_test_command(
        "./run{{#files}} {{value}}{{/files}}",
        ["tests/test_alpha.py", "tests/test_beta.py"],
    ) == "./run tests/test_alpha.py tests/test_beta.py"
    assert render_test_command(
        "./gradlew{{#fqclasses}} --tests {{value}}{{/fqclasses}}",
        ["src/test/java/com/example/FooTest.java", "pkg/foo_test.go"],
    ) == "./gradlew --tests com.example.FooTest --tests foo_test"


def test_render_test_command_supports_filter_passing_tests_markers() -> None:
    assert render_test_command(
        "./gradlew{{#moduletests}} {{value}}{{/moduletests}}",
        ["server/src/test/java/com/example/FooTest.java"],
    ) == "./gradlew server:test --tests FooTest"
    assert render_test_command(
        "./run{{#packages}} {{value}}{{/packages}}",
        ["pkg/a/test_one.py", "pkg/b/test_two.py"],
    ) == "./run pkg/a pkg/b"


def test_render_test_command_rejects_multiple_marker_kinds() -> None:
    try:
        render_test_command(
            "./run{{#files}} {{value}}{{/files}}{{#classes}} {{value}}{{/classes}}",
            ["tests/test_alpha.py"],
        )
    except ValueError as exc:
        assert "at most one" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected mixed markers to fail")


def test_build_testsome_command_falls_back_to_project_properties(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "project"
    props_dir = project / ".brokk"
    props_dir.mkdir(parents=True)
    (props_dir / "project.properties").write_text(
        'buildDetailsJson={"testSomeCommand":"./gradlew{{#classes}} --tests {{value}}{{/classes}}"}\n',
        encoding="utf-8",
    )

    cmd = build_testsome_command(project, {}, ["src/test/java/com/example/FooTest.java"])

    assert cmd == "./gradlew --tests FooTest"

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Mapping

import pystache


def _get_module_from_path(path: str) -> str | None:
    """Extract Gradle module from file path."""
    match = re.match(r"^([^/]+)/src/", path)
    return match.group(1) if match else None


def _to_fq_class(path_str: str) -> str:
    """Convert a file path to a fully-qualified Java/Kotlin/Groovy class name."""
    s = path_str.replace("\\", "/")
    if not (s.endswith(".java") or s.endswith(".kt") or s.endswith(".groovy")):
        return pathlib.Path(path_str).stem

    s_noext = re.sub(r"\.(java|kt|groovy)$", "", s)

    patterns = [
        r"/src/test/(java|kotlin|groovy)/",
        r"/src/[^/]+/(java|kotlin|groovy)/",
        r"/test/(java|kotlin|groovy)/",
        r"/(java|kotlin|groovy)/",
    ]
    rel = None
    for pat in patterns:
        match = re.search(pat, s_noext)
        if match:
            rel = s_noext[match.end():]
            break
    if rel is None:
        rel = s_noext

    return rel.replace("/", ".")


def _to_crate(path_str: str) -> str:
    """Extract Rust crate name from a file path."""
    parts = pathlib.Path(path_str).parts
    for index, part in enumerate(parts):
        if part == "crates" and index + 1 < len(parts):
            return parts[index + 1]
    return pathlib.Path(path_str).parts[0]


def _make_template_list(items: list[str]) -> list[dict[str, object]]:
    return [
        {"value": value, "first": index == 0, "last": index == len(items) - 1}
        for index, value in enumerate(items)
    ]


def resolve_testsome_template(project_path: pathlib.Path, env: Mapping[str, str]) -> str | None:
    template = env.get("BRK_TESTSOME_CMD")
    if template:
        return template

    properties_file = project_path / ".brokk" / "project.properties"
    build_details_json_str: str | None = None
    if properties_file.exists():
        with open(properties_file, "r", encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("buildDetailsJson="):
                    build_details_json_str = stripped.split("=", 1)[1].strip()
                    build_details_json_str = build_details_json_str.replace(r"\:", ":")
                    break
    if build_details_json_str is None:
        return None

    try:
        build_details = json.loads(build_details_json_str)
    except json.JSONDecodeError:
        return None

    template = build_details.get("testSomeCommand")
    return template if isinstance(template, str) and template else None


def render_test_command(template: str, test_files: list[str]) -> str:
    is_moduletests_based = "{{#moduletests}}" in template
    is_packages_based = "{{#packages}}" in template
    is_files_based = "{{#files}}" in template
    is_fq_based = "{{#fqclasses}}" in template
    is_classes_based = "{{#classes}}" in template
    is_crates_based = "{{#crates}}" in template

    if (
        sum(
            1
            for flag in (
                is_moduletests_based,
                is_packages_based,
                is_files_based,
                is_fq_based,
                is_classes_based,
                is_crates_based,
            )
            if flag
        )
        > 1
    ):
        raise ValueError(
            "Template must contain at most one of #moduletests, #packages, #classes, "
            "#fqclasses, #files, or #crates"
        )

    if not any(
        (
            is_moduletests_based,
            is_packages_based,
            is_files_based,
            is_fq_based,
            is_classes_based,
            is_crates_based,
        )
    ):
        return template

    if is_moduletests_based:
        items = []
        for path in test_files:
            module = _get_module_from_path(path)
            class_name = pathlib.Path(path).stem
            if module:
                items.append(f"{module}:test --tests {class_name}")
        key = "moduletests"
        values = sorted(set(items))
    elif is_packages_based:
        key = "packages"
        values = sorted({str(pathlib.Path(path).parent) for path in test_files})
    elif is_files_based:
        key = "files"
        values = list(test_files)
    elif is_fq_based:
        key = "fqclasses"
        values = sorted(
            {
                _to_fq_class(path)
                for path in test_files
                if path.endswith((".java", ".kt", ".groovy", ".go", ".cs"))
            }
        )
    elif is_crates_based:
        key = "crates"
        values = sorted({_to_crate(path) for path in test_files})
    else:
        key = "classes"
        values = sorted(
            {
                pathlib.Path(path).stem
                for path in test_files
                if path.endswith((".java", ".kt", ".groovy", ".go", ".cs"))
            }
        )

    context = {key: _make_template_list(values)}
    return pystache.render(template, context)


def build_testsome_command(project_path: pathlib.Path, env: Mapping[str, str], test_files: list[str]) -> str:
    template = resolve_testsome_template(project_path, env)
    if not template:
        raise ValueError("No test command available for strategy some")
    return render_test_command(template, test_files)

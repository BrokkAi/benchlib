import json
import os
import pathlib
import re
import subprocess
import sys
import threading

CLI_CMD: list[str] | None = None

CACHE_DIR_VARS = [
    "BRK_WORKTREE_ROOT",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "GOCACHE",
    "GOMODCACHE",
    "GOTMPDIR",
    "GRADLE_USER_HOME",
    "npm_config_cache",
    "PIP_CACHE_DIR",
    "NUGET_PACKAGES",
    "DOTNET_CLI_HOME",
    "TMPDIR",
]

EMBEDDED_PATH_VARS = {
    "MAVEN_OPTS": r"-Dmaven\.repo\.local=(\S+)",
    "SBT_OPTS": r"-Dsbt\.global\.base=(\S+)|-Dsbt\.ivy\.home=(\S+)",
}


def set_cli_bin(path: pathlib.Path, cli_jar: pathlib.Path | None = None) -> None:
    """Configure the Brokk CLI command prefix."""
    global CLI_CMD
    cli_path = path / "cli" if path.is_dir() else path
    CLI_CMD = [str(cli_path)]
    if cli_jar is not None:
        CLI_CMD.extend(["--jar", str(cli_jar)])


def get_cli_command() -> list[str]:
    if CLI_CMD is not None:
        return list(CLI_CMD)

    cli_path = pathlib.Path(os.getenv("BRK_CLI_BIN", "../brokk/cli"))
    return [str(cli_path)]


def validate_api_key() -> None:
    if os.getenv("BROKK_API_KEY"):
        return

    props_path = pathlib.Path.home() / ".config" / "brokk" / "brokk.properties"
    if props_path.exists():
        with open(props_path, "r", encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if stripped.startswith("brokkApiKey="):
                    key = stripped.split("=", 1)[1].strip()
                    if key:
                        os.environ["BROKK_API_KEY"] = key
                        return

    print("Error: BROKK_API_KEY environment variable is required.", file=sys.stderr)
    print("Alternatively, set brokkApiKey in ~/.config/brokk/brokk.properties", file=sys.stderr)
    sys.exit(1)


def validate_cache_dirs() -> None:
    paths_to_create: list[tuple[str, str]] = []

    for var in CACHE_DIR_VARS:
        value = os.getenv(var)
        if value:
            paths_to_create.append((var, value))

    for var, pattern in EMBEDDED_PATH_VARS.items():
        value = os.getenv(var)
        if not value:
            continue
        for match in re.finditer(pattern, value):
            for group in match.groups():
                if group:
                    paths_to_create.append((var, group))

    for var_name, path in paths_to_create:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            print(f"Error: Cannot create cache directory for {var_name}", file=sys.stderr)
            print(f"  Path: {path}", file=sys.stderr)
            print(f"  Reason: {exc}", file=sys.stderr)
            sys.exit(1)


def _get_cli_version() -> str:
    try:
        result = subprocess.run(
            [*get_cli_command(), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _get_cli_models_json() -> list:
    env_val = os.getenv("BROKK_FAVORITE_MODELS")
    if env_val:
        try:
            return json.loads(env_val)
        except json.JSONDecodeError:
            print(f"Error: Cannot parse BROKK_FAVORITE_MODELS: {env_val}", file=sys.stderr)
            sys.exit(1)

    try:
        result = subprocess.run(
            [*get_cli_command(), "--list-models"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"Error: Failed to run CLI for model validation: {exc}", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        print(f"Error: CLI --list-models failed (exit {result.returncode}):", file=sys.stderr)
        print(((result.stdout or "") + (result.stderr or "")).strip(), file=sys.stderr)
        sys.exit(1)

    stdout_lines = result.stdout.strip().splitlines()
    json_line = stdout_lines[-1] if stdout_lines else ""
    try:
        return json.loads(json_line)
    except json.JSONDecodeError:
        print(f"Error: Cannot parse --list-models output: {json_line}", file=sys.stderr)
        sys.exit(1)


def _read_proxy_setting() -> str:
    env_val = os.getenv("BROKK_PROXY")
    if env_val:
        return env_val

    props_path = pathlib.Path.home() / ".config" / "brokk" / "brokk.properties"
    if props_path.exists():
        with open(props_path, "r", encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if stripped.startswith("llmProxySetting="):
                    return stripped.split("=", 1)[1].strip()
    return "BROKK"


_cli_info_cache: dict | None = None
_cli_info_lock = threading.Lock()


def get_cli_info() -> dict:
    global _cli_info_cache
    if _cli_info_cache is not None:
        return _cli_info_cache

    with _cli_info_lock:
        if _cli_info_cache is not None:
            return _cli_info_cache
        _cli_info_cache = {
            "cliVersion": _get_cli_version(),
            "proxy": _read_proxy_setting(),
            "favoriteModels": _get_cli_models_json(),
        }
        return _cli_info_cache


def validate_models(models: list[str]) -> None:
    info = get_cli_info()
    try:
        available = {entry["alias"] for entry in info["favoriteModels"]}
    except KeyError:
        print("Error: Cannot parse model aliases from CLI output", file=sys.stderr)
        sys.exit(1)

    invalid = [model for model in models if model not in available]
    if invalid:
        print(f"Error: Unknown model alias(es): {', '.join(invalid)}", file=sys.stderr)
        print(f"Available aliases: {', '.join(sorted(available))}", file=sys.stderr)
        sys.exit(1)


def run_cli(
    cmd: list[str],
    log_file: pathlib.Path,
    env: dict | None = None,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess:
    debug = bool(os.getenv("BB_DEBUG"))
    if debug:
        print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        if env and "JAVA_HOME" in env:
            print(f"Using JAVA_HOME: {env['JAVA_HOME']}", file=sys.stderr)

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    ) as proc, open(log_file, "ab") as log_fp:
        def _forward_output() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_fp.write(line)
                log_fp.flush()
                if not debug:
                    continue
                try:
                    sys.stderr.buffer.write(line)
                    sys.stderr.flush()
                except AttributeError:
                    sys.stderr.write(line.decode(errors="replace"))
                    sys.stderr.flush()

        output_thread = threading.Thread(target=_forward_output, daemon=True)
        output_thread.start()
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            output_thread.join()
            raise
        output_thread.join()

    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=None, stderr=None)

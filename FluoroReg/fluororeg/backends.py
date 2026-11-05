"""Frozen backend discovery and explicit, out-of-tree runtime setup.

No helper installs software on import, discovery, status, or registration. Only
``setup_tools`` installs environments. xreg remains a separately configured tool.
"""

import json
import os
import shutil
from pathlib import Path
import subprocess

from .runner import child_env, new_log_path, run_tool
from .status import InputError, ToolError, emit

COMPONENT_KEYS = {"pose": "pose_dir", "annotation": "annotation_3d_dir"}
REQUIRED_VERSIONS = {
    "pose": {"torch": "2.10.0", "diffdrr": "0.6.0"},
    "annotation": {"TotalSegmentator": "2.7.0", "torch": "2.3.1", "numpy": "1.26.4",
                   "scipy": "1.14.0", "nibabel": "5.2.1", "nnunetv2": "2.4.2", "pyvista": "0.46.3"},
}
MARKER = "fluororeg-runtime.json"


def bundled_tools_root(module_file=None):
    """Return checkout/installed tools root, or None; never search sibling checkouts."""
    origin = Path(module_file or __file__).resolve()
    for ancestor in origin.parents:
        candidates = (ancestor / "tools", ancestor / "share" / "Slicer-5.8" / "SlicerFluoroReg" / "tools")
        for candidate in candidates:
            if (candidate / "VENDORED_COMPONENTS.json").is_file():
                return str(candidate)
    return None


def component_info(component, tools_root=None):
    """Manifest component fields plus absolute ``source_dir``; missing payload is an input error."""
    if component not in COMPONENT_KEYS:
        raise InputError(f"unknown backend component {component}")
    root = tools_root or bundled_tools_root()
    if not root:
        raise InputError("bundled backend sources are missing; reinstall SlicerFluoroReg or set --pose-dir / "
                         "--annotation-3d-dir explicitly")
    try:
        with open(os.path.join(root, "VENDORED_COMPONENTS.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported schema_version")
        info = dict(manifest["components"][component])
        directory = info["directory"]
        digest = info["snapshot_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid snapshot_sha256")
        source = os.path.abspath(os.path.join(root, directory))
        if os.path.commonpath((os.path.abspath(root), source)) != os.path.abspath(root):
            raise ValueError("component directory escapes tools root")
        info["source_dir"] = source
        return info
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InputError(f"invalid bundled backend manifest: {exc}") from exc


def bundled_defaults():
    """Default source paths only; an unavailable distribution never falls back to siblings."""
    if not bundled_tools_root():
        return {}
    return {key: component_info(component)["source_dir"] for component, key in COMPONENT_KEYS.items()}


def runtime_paths(component, snapshot_sha256, environ=None):
    """Versioned env and shared model caches; no directories are created by discovery."""
    env = os.environ if environ is None else environ
    home = env.get("HOME") or os.path.expanduser("~")
    data = env.get("XDG_DATA_HOME", "")
    cache = env.get("XDG_CACHE_HOME", "")
    if not os.path.isabs(data):
        data = os.path.join(home, ".local", "share")
    if not os.path.isabs(cache):
        cache = os.path.join(home, ".cache")
    return {"env_dir": os.path.join(data, "fluororeg", "backends", component, snapshot_sha256, "env"),
            "models_dir": os.path.abspath(env.get("BIGSS_POSE_WEIGHTS_DIR") or
                                         os.path.join(cache, "fluororeg", "models")),
            "totalsegmentator_dir": os.path.abspath(env.get("TOTALSEG_HOME_DIR") or
                                                  os.path.join(cache, "fluororeg", "totalsegmentator"))}


def backend_spec(settings, component):
    """Resolve source, runtime paths and bundled identity for one component."""
    source = settings.require(COMPONENT_KEYS[component])
    info = component_info(component) if bundled_tools_root() else None
    bundled = bool(info and os.path.realpath(source) == os.path.realpath(info["source_dir"]))
    # External pose checkouts retain their standalone environment; annotation needs an explicit Conda target.
    paths = runtime_paths(component, info["snapshot_sha256"] if bundled else "external")
    explicit_env = settings.get("pose_env_dir" if component == "pose" else "annotation_3d_env_prefix")
    if component == "pose" and not explicit_env:
        explicit_env = os.environ.get("BIGSS_POSE_ENV_DIR")
    if explicit_env:
        explicit_env = os.path.abspath(explicit_env)
    if component == "annotation" and not bundled and not explicit_env and not settings.get("annotation_3d_env"):
        raise InputError("external annotation sources require --annotation-3d-env NAME or "
                         "--annotation-3d-env-prefix DIR")
    env_dir = explicit_env or (paths["env_dir"] if bundled or component == "annotation"
                               else os.path.join(source, ".venv"))
    return dict(paths, source_dir=source, env_dir=env_dir, bundled=bundled,
                snapshot_sha256=info["snapshot_sha256"] if bundled else None,
                managed=bundled and not explicit_env and not (component == "annotation" and settings.get("annotation_3d_env")))


def backend_env(settings, component, spec=None):
    spec = spec or backend_spec(settings, component)
    extra = {"CUDA_VISIBLE_DEVICES": settings.get("gpu") or "0", "PYTHONNOUSERSITE": "1",
             "BIGSS_POSE_WEIGHTS_DIR": spec["models_dir"],
             "TOTALSEG_HOME_DIR": spec["totalsegmentator_dir"]}
    if component == "pose":
        extra["BIGSS_POSE_ENV_DIR"] = spec["env_dir"]
    return child_env(extra=extra)


def setup_instruction(component):
    return f"python -m fluororeg tools setup --component {component}"


def _probe_environment(component, spec, env, python_argv=None):
    python = os.path.join(spec["env_dir"], "bin", "python")
    if python_argv is None and not os.path.isfile(python):
        return "environment is missing"
    code = ("import json,sys,importlib.metadata as m; packages="
            + repr(list(REQUIRED_VERSIONS[component]))
            + "; print(json.dumps({'python':list(sys.version_info[:3]),"
              "'versions':{k:m.version(k) for k in packages}}))")
    try:
        result = subprocess.run([*(python_argv or [python]), "-c", code], cwd=spec["source_dir"], env=env,
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            return "environment dependencies are missing or unreadable: " + result.stderr.strip()[-1000:]
        record = json.loads(result.stdout)
        version = record["python"]
        expected = [3, 10] if component == "pose" else [3, 12, 4]
        if version[:len(expected)] != expected:
            return f"wrong Python version {'.'.join(map(str, version))}; expected {'.'.join(map(str, expected))}"
        for package, expected_version in REQUIRED_VERSIONS[component].items():
            actual = record["versions"].get(package, "")
            if actual.split("+")[0] != expected_version:
                return f"wrong {package} version {actual}; expected {expected_version}"
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        return f"cannot inspect environment: {exc}"
    return None


def environment_problem(settings, component, spec=None, probe=True):
    spec = spec or backend_spec(settings, component)
    if component == "annotation" and settings.get("annotation_3d_env"):
        annotation_conda_args(settings, spec)
        if not probe or not spec["bundled"]:
            return None  # Explicit external checkouts retain their native runtime contract.
        env = backend_env(settings, component, spec)
        conda = settings.require_executable("conda", env.get("PATH"))
        return _probe_environment(component, spec, env,
                                  [conda, "run", *annotation_conda_args(settings, spec),
                                   "--no-capture-output", "python"])
    if not spec["bundled"] and not settings.get("pose_env_dir" if component == "pose" else "annotation_3d_env_prefix"):
        return None  # Intentional external checkout retains its native launcher contract.
    if spec["managed"]:
        try:
            with open(os.path.join(spec["env_dir"], MARKER), encoding="utf-8") as handle:
                marker = json.load(handle)
            if marker.get("snapshot_sha256") != spec["snapshot_sha256"]:
                return "environment belongs to a different source snapshot"
        except (OSError, ValueError, AttributeError):
            return "environment has not been set up for this source snapshot"
    if component == "annotation" and not os.path.isdir(os.path.join(spec["env_dir"], "conda-meta")):
        return "Conda environment prefix is missing or invalid"
    if not os.path.isfile(os.path.join(spec["env_dir"], "bin", "python")):
        return "environment is missing"
    return _probe_environment(component, spec, backend_env(settings, component, spec)) if probe else None


def require_environment(settings, component, spec=None):
    spec = spec or backend_spec(settings, component)
    problem = environment_problem(settings, component, spec)
    if problem:
        location = settings.get("annotation_3d_env") if component == "annotation" and settings.get("annotation_3d_env") else spec["env_dir"]
        raise InputError(f"{component}: {problem} at {location}; run `{setup_instruction(component)}` "
                         "with the same tool/settings overrides")
    return spec


def annotation_conda_args(settings, spec=None):
    if settings.get("annotation_3d_env") and settings.get("annotation_3d_env_prefix"):
        raise InputError("choose --annotation-3d-env (name) or --annotation-3d-env-prefix, not both")
    if settings.get("annotation_3d_env"):
        return ["-n", settings.get("annotation_3d_env")]
    spec = spec or backend_spec(settings, "annotation")
    return ["-p", spec["env_dir"]]


def tools_status(settings, component="all", probe=True):
    components = COMPONENT_KEYS if component == "all" else (component,)
    result = {}
    for name in components:
        try:
            spec = backend_spec(settings, name)
            if name == "annotation":
                annotation_conda_args(settings, spec)
            problem = environment_problem(settings, name, spec, probe=probe)
            result[name] = dict(spec, ready=not problem, problem=problem,
                                setup_command=setup_instruction(name))
        except InputError as exc:
            result[name] = {"ready": False, "problem": str(exc), "setup_command": setup_instruction(name)}
    return result


def _configure_annotation_cache(spec, env, python_argv):
    """Keep upstream usage reporting disabled in the selected external cache."""
    code = ("from totalsegmentator.config import get_totalseg_dir,setup_totalseg,set_config_key;"
            "get_totalseg_dir().mkdir(parents=True,exist_ok=True);"
            "setup_totalseg();set_config_key('send_usage_stats',False)")
    log = new_log_path(os.path.dirname(spec["env_dir"]), "annotation", "cache")
    rc = run_tool([*python_argv, "-c", code], cwd=spec["source_dir"], log_path=log, env=env)
    if rc:
        raise ToolError(f"annotation cache configuration exited with status {rc}; see {log}", {"log": log})


def setup_tools(settings, component="all"):
    """Explicit operator action. Run native setup out of Slicer and log outside the extension."""
    components = COMPONENT_KEYS if component == "all" else (component,)
    result = {}
    for name in components:
        spec = backend_spec(settings, name)
        env = backend_env(settings, name, spec)
        env["PIP_NO_INPUT"] = "1"
        log = new_log_path(os.path.dirname(spec["env_dir"]), name, "setup")
        if name == "pose":
            script = os.path.join(spec["source_dir"], "scripts", "setup_env.sh")
            if not os.path.isfile(script):
                raise InputError(f"pose source has no scripts/setup_env.sh: {spec['source_dir']}")
            argv = ["bash", script]
        else:
            conda = settings.require_executable("conda", env.get("PATH"))
            target = annotation_conda_args(settings, spec)
            environment = os.path.join(spec["source_dir"], "environment.yml")
            if not os.path.isfile(environment):
                raise InputError(f"annotation source has no pinned environment.yml: {spec['source_dir']}")
            # Conda creates its pip requirements beside the YAML; installed source may be read-only.
            staged_environment = os.path.splitext(log)[0] + ".environment.yml"
            shutil.copyfile(environment, staged_environment)
            environment = staged_environment
            exists = os.path.isdir(os.path.join(spec["env_dir"], "conda-meta")) if target[0] == "-p" else True
            argv = [conda, "env", "update" if exists else "create", *target, "--file", environment]
        if spec["managed"]:
            marker = os.path.join(spec["env_dir"], MARKER)
            if os.path.isfile(marker):
                os.unlink(marker)
        emit("progress", "tools", component=name, message=f"setting up {name}", log=log)
        rc = run_tool(argv, cwd=spec["source_dir"], log_path=log, env=env)
        if rc:
            raise ToolError(f"{name} setup exited with status {rc}; see {log}", {"log": log})
        # Verify the actual interpreter/dependencies before declaring a managed runtime ready.
        python_argv = ([conda, "run", *target, "--no-capture-output", "python"]
                       if name == "annotation" and settings.get("annotation_3d_env") else None)
        problem = _probe_environment(name, spec, env, python_argv)
        if problem:
            raise ToolError(f"{name} setup did not produce the required runtime: {problem}; see {log}")
        if name == "annotation":
            _configure_annotation_cache(spec, env, python_argv or [os.path.join(spec["env_dir"], "bin", "python")])
        if spec["managed"]:
            with open(os.path.join(spec["env_dir"], MARKER), "w", encoding="utf-8") as handle:
                json.dump({"snapshot_sha256": spec["snapshot_sha256"]}, handle)
                handle.write("\n")
        result[name] = dict(spec, ready=True, log=log)
    return result

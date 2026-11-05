"""Machine-specific tool paths: never stored in a project.

Precedence per key: command-line flag > ``--settings FILE`` > the user file
``~/.config/fluororeg/settings.json`` (or ``$FLUOROREG_SETTINGS``) > built-in default.
The Slicer module reads and writes the same user file.
"""

import json
import os
import shutil

from .backends import bundled_defaults
from .status import InputError

KEYS = ("xreg_dir", "xreg_build_dir", "annotation_3d_dir", "conda", "annotation_3d_env", "pose_dir", "gpu",
        "xray_template", "pose_env_dir", "annotation_3d_env_prefix")
# Keys the settings file may also hold for the shell scripts; they are not pipeline flags.
SCRIPT_KEYS = ("slicer",)
FILE_KEYS = KEYS + SCRIPT_KEYS
DEFAULTS = {"conda": "conda", "gpu": "0"}
PATH_KEYS = ("xreg_dir", "xreg_build_dir", "annotation_3d_dir", "pose_dir", "pose_env_dir",
             "annotation_3d_env_prefix")
FILE_PATH_KEYS = ("xray_template",)


def flag_of(key):
    return "--" + key.replace("_", "-")


def user_settings_path():
    override = os.environ.get("FLUOROREG_SETTINGS")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(os.path.expanduser("~"), ".config", "fluororeg", "settings.json")


def read_user_settings(path=None):
    """{key: value} stored in the user settings file ({} when it does not exist)."""
    return _read_settings_file(path or user_settings_path(), required=False)


def write_user_settings(values, path=None):
    """Sets the given keys in the user settings file (an empty value removes the key); returns the file."""
    path = path or user_settings_path()
    data = read_user_settings(path)
    for key, value in values.items():
        if key not in FILE_KEYS:
            raise InputError(f"unknown settings key {key}; known: {', '.join(FILE_KEYS)}")
        value = "" if value is None else str(value).strip()
        if value:
            data[key] = value
        else:
            data.pop(key, None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({k: data[k] for k in FILE_KEYS if k in data}, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)
    return path


def _read_settings_file(path, required):
    if not os.path.isfile(path):
        if required:
            raise InputError(f"settings file not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except ValueError as exc:
        raise InputError(f"settings file {path} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise InputError(f"settings file {path} must hold a JSON object")
    unknown = sorted(set(data) - set(FILE_KEYS))
    if unknown:
        raise InputError(f"settings file {path}: unknown key(s) {', '.join(unknown)}; known: {', '.join(FILE_KEYS)}")
    return {k: str(v) for k, v in data.items() if v is not None and str(v) != ""}


class Settings:
    def __init__(self, values):
        self._values = dict(values)

    @classmethod
    def resolve(cls, flags, settings_file=None, user_file=None):
        """``flags``: {key: value or None}; ``settings_file``: the ``--settings`` path or None."""
        values = dict(DEFAULTS, **bundled_defaults())
        values.update(_read_settings_file(user_file or user_settings_path(), required=False))
        if settings_file:
            values.update(_read_settings_file(settings_file, required=True))
        values.update({k: str(v) for k, v in (flags or {}).items() if k in KEYS and v is not None and str(v) != ""})
        values = {k: v for k, v in values.items() if k in KEYS}
        for key in PATH_KEYS + FILE_PATH_KEYS:
            if key in values:
                values[key] = os.path.abspath(os.path.expanduser(values[key]))
        if "xreg_build_dir" not in values and "xreg_dir" in values:
            values["xreg_build_dir"] = os.path.join(values["xreg_dir"], "build")
        return cls(values)

    def get(self, key):
        return self._values.get(key)

    def require(self, key):
        value = self._values.get(key)
        if not value:
            if key == "xreg_build_dir":
                raise InputError("missing --xreg-dir or --xreg-build-dir (or 'xreg_dir' in the settings file)")
            raise InputError(f"missing {flag_of(key)} (or '{key}' in the settings file)")
        if key in PATH_KEYS and not os.path.isdir(value):
            raise InputError(f"{flag_of(key)} {value} is not a directory")
        if key in FILE_PATH_KEYS and not os.path.isfile(value):
            raise InputError(f"{flag_of(key)} {value} is not a file")
        return value

    def require_executable(self, key, env_path):
        """``conda``: a path or a name looked up on the child PATH."""
        value = self.require(key)
        found = shutil.which(value, path=env_path)
        if not found:
            raise InputError(f"{flag_of(key)} {value} is not an executable (not found on PATH)")
        return found

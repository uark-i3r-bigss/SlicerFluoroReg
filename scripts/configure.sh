#!/usr/bin/env bash
# Configure operator paths; frozen pose/annotation sources are discovered by fluororeg.settings.
# No backend is installed or downloaded here. Run scripts/setup_tools.sh explicitly afterwards.
# Env: FLUOROREG_PYTHON (PythonSlicer or python3), FLUOROREG_SETTINGS (settings file).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
"$py" - "$@" <<'EOF'
import argparse
import glob
import json
import os
import shutil
from fluororeg.settings import Settings, read_user_settings, write_user_settings
from fluororeg.status import InputError

parser = argparse.ArgumentParser(description="Configure FluoroReg paths without installing backends.")
parser.add_argument("--slicer", help="Slicer executable (otherwise preserve settings or discover locally)")
parser.add_argument("--xreg", dest="xreg_dir", help="manually installed matching xreg source directory")
parser.add_argument("--xreg-build-dir", help="manually installed matching xreg build directory")
parser.add_argument("--annotation-3d", dest="annotation_3d_dir", help="explicit annotation source override")
parser.add_argument("--pose", dest="pose_dir", help="explicit pose source override")
parser.add_argument("--conda", help="Conda executable (default: conda on PATH)")
parser.add_argument("--annotation-3d-env", help="explicit named annotation Conda environment override")
parser.add_argument("--annotation-3d-env-prefix", help="explicit annotation environment prefix override")
parser.add_argument("--pose-env-dir", help="explicit pose environment directory override")
parser.add_argument("--gpu", help="GPU index (default: 0)")
parser.add_argument("--xray-template", help="authorized X-ray DICOM geometry template")
args = parser.parse_args()
values = {key: value for key, value in vars(args).items() if value is not None}
try:
    resolved = Settings.resolve(values)
    for key, marker in (("pose_dir", "scripts/pose.sh"),
                        ("annotation_3d_dir", "process_medical_scans.py")):
        directory = resolved.require(key)
        if not os.path.isfile(os.path.join(directory, marker)):
            raise InputError(f"{key}: {directory} has no {marker}")
        print(f"OK       {key}: {directory}")
    for key in ("xreg_dir", "xreg_build_dir", "xray_template"):
        if resolved.get(key):
            resolved.require(key)
    if resolved.get("gpu") and not resolved.get("gpu").isdigit():
        raise InputError("gpu must be a nonnegative integer")
    if "xray_template" in values:
        with open(resolved.require("xray_template"), "rb") as handle:
            handle.seek(128)
            if handle.read(4) != b"DICM":
                raise InputError("xray-template must be a DICOM file")
    slicer = values.get("slicer") or read_user_settings().get("slicer")
    if not slicer:
        candidates = [os.environ.get("SLICER"), shutil.which("Slicer")]
        candidates += sorted(glob.glob(os.path.expanduser("~/Slicer*/Slicer")))
        candidates += sorted(glob.glob("/opt/Slicer*/Slicer"))
        slicer = next((p for p in candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)
    if slicer:
        if not os.path.isfile(slicer) or not os.access(slicer, os.X_OK):
            raise InputError(f"slicer is not executable: {slicer}")
        values["slicer"] = os.path.realpath(slicer)
    for key in ("xreg_dir", "xreg_build_dir", "annotation_3d_dir", "pose_dir",
                "pose_env_dir", "annotation_3d_env_prefix", "xray_template"):
        if key in values:
            values[key] = os.path.abspath(os.path.expanduser(values[key]))
    path = write_user_settings(values)
except (InputError, OSError) as exc:
    parser.exit(1, f"Nothing written: {exc}\n")
print(f"Wrote {path}:")
print(json.dumps(read_user_settings(path), indent=2))
print("Next: scripts/setup_tools.sh (explicit network/environment setup); xreg remains a manual install.")
EOF

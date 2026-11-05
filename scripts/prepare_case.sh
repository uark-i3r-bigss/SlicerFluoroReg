#!/usr/bin/env bash
# Prepares one case of an experiment folder for registration: init, import-ct, annotate-3d (stages the CT,
# segmentation, models and 3D landmarks), import-xrays (converts them to projection H5) and pose train
# (creates the pose config when missing). Initial poses are predicted by register.sh. Safe to re-run: a step
# whose output exists is skipped.
#
#   scripts/prepare_case.sh --project P --ct PATH --xrays PATH [PATH ...] [--case C] [--gpu N] [--redo-pose]
#
#   --project P       experiment folder (created when missing)
#   --ct PATH         CT as a .nii/.nii.gz file or a DICOM folder
#   --xrays PATH ...  X-ray DICOM or image files, or folders of them (ids 001, 002, ... in the given order,
#                     a folder's files in natural file-name order); images need "xray_template" in the
#                     settings file (scripts/configure.sh --xray-template FILE)
#   --case C          case id (default: the CT file stem or folder name)
#   --gpu N           GPU index for this run (default: "gpu" of the settings file)
#   --redo-pose       train again, e.g. after editing configs/pose_estimation/<case>.json
#
# Tool paths come from the settings file (scripts/configure.sh).
# Env: FLUOROREG_PYTHON (default: Slicer's PythonSlicer, else python3), FLUOROREG_SETTINGS.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"

project="" ct="" case="" redo_pose=0
xrays=()
tools=()
need() { if [ $# -lt 2 ] || [ -z "$2" ]; then echo "$1 needs a value" >&2; exit 2; fi; }
while [ $# -gt 0 ]; do
    case "$1" in
        --project) need "$@"; project="$2"; shift ;;
        --ct) need "$@"; ct="$2"; shift ;;
        --xrays)
            need "$@"
            while [ $# -gt 1 ] && [[ "$2" != --* ]]; do xrays+=("$2"); shift; done ;;
        --case) need "$@"; case="$2"; shift ;;
        --gpu) need "$@"; tools=(--gpu "$2"); shift ;;
        --redo-pose) redo_pose=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
if [ -z "$project" ] || [ -z "$ct" ] || [ ${#xrays[@]} -eq 0 ]; then
    echo "--project, --ct and --xrays are required (see --help)" >&2; exit 2
fi
for x in "${xrays[@]}"; do
    if [ ! -e "$x" ]; then echo "--xrays $x does not exist" >&2; exit 2; fi
done
project="$(realpath -m "$project")"
# Through the environment: the PythonSlicer launcher prints its usage (to stdout) for an empty argument.
case="$(CT="$ct" CASE="$case" "$py" -c 'import os, sys
from fluororeg.ct import ct_input
from fluororeg.status import InputError
try:
    print(ct_input(os.environ["CT"], os.environ["CASE"] or None)[3])
except InputError as exc:
    sys.exit(f"--ct: {exc}")')"

summary=()
ran() { summary+=("ran      $1"); }
skipped() { summary+=("skipped  $1 ($2)"); echo "= skip $1: $2"; }

echo "Project $project, case $case"
if [ -f "$project/project_manifest.json" ]; then
    skipped init "project_manifest.json exists"
else
    pipeline init --project "$project"; ran init
fi

if [ -f "$project/meta_data/volumes/${case}_ct.nii.gz" ]; then
    skipped import-ct "staged meta_data/volumes/${case}_ct.nii.gz exists"
    skipped annotate-3d "case is staged"
else
    pipeline import-ct --project "$project" --input "$ct" --case "$case"; ran import-ct
    pipeline annotate-3d --project "$project" --case "$case" "${tools[@]}"; ran annotate-3d
fi

if [ -f "$project/meta_data/XrayDICOM/xray_index.json" ]; then
    skipped import-xrays "meta_data/XrayDICOM/xray_index.json exists"
    pipeline convert-xrays --project "$project" "${tools[@]}"
    converted="$(echo "$result" | json_get 'len(r["converted"])')"
    if [ "$converted" = "0" ]; then
        skipped convert-xrays "every H5 exists"
    else
        ran "convert-xrays ($converted converted)"
    fi
else
    pipeline import-xrays --project "$project" --input "${xrays[@]}" "${tools[@]}"; ran "import-xrays (converted)"
fi

if [ "$redo_pose" -eq 0 ] && [ -f "$project/meta_data/pose_models/$case/active.json" ]; then
    skipped "pose train" "an active model exists; --redo-pose retrains"
else
    rc=0
    pipeline pose train --project "$project" --case "$case" "${tools[@]}" || rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then exit "$rc"; fi
    ran "pose train"
fi

echo
echo "Steps:"
printf '  %s\n' "${summary[@]}"
STATUS_OUTPUT="$("$py" -m fluororeg status --project "$project")" "$py" - "$case" <<'EOF'
import json, os, sys
case = sys.argv[1]
status = None
for line in os.environ["STATUS_OUTPUT"].splitlines():
    if line.startswith("FLUOROREG "):
        event = json.loads(line[len("FLUOROREG "):])
        if event.get("event") == "result":
            status = event
xrays = status["xrays"]
print(f"X-rays: {len(xrays['ids'])} imported, {len(xrays['converted'])} converted")
model = (status["pose"].get(case) or {}).get("model") or {}
if not model.get("active"):
    print("Pose model: none active (training did not pass its self-check?)")
    sys.exit(0)
capture = model.get("self_check_capture")
capture = f"{capture:.0%}" if isinstance(capture, (int, float)) else "n/a"
print(f"Pose model: {model['name']}, self-check {capture}")
print("Next: scripts/register.sh (predicts the initial poses, then registers)")
EOF

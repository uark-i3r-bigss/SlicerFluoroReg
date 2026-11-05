#!/usr/bin/env bash
# Predicts the initial poses of the chosen X-rays with the case's pose model, runs one xreg registration of an
# experiment folder and its assessment, then prints the run directory and the counts of results with high /
# medium / low certainty of correct registration.
#
#   scripts/register.sh --project P [--case C] (--config FILE | --app APP [--ids ID ...] [--name N]
#                       [--femur left|right]) [--no-predict] [--replay]
#
#   --config FILE  existing xreg config (configs/xreg/<name>.json), run as it is
#   --app APP      single-target, single-target-multiview, multi-target or multi-target-multiview: uses
#                  configs/xreg/<name>.json, created from the default when missing, else --ids are set on it
#   --ids ID ...   X-ray ids (one for single-view apps, several for multiview)
#   --name N       config name (default: <app>_<case>[_femur-<side>], dashes as underscores in the app)
#   --femur SIDE   femur of the multi-target apps
#   --case C       case (default: the only case of the project)
#   --no-predict   use the initial poses already in meta_data/initial_poses/<case>
#   --replay       also render the optimizer replay videos (off by default)
#
# Configs are copied once into configs/xreg/ and edited by hand afterwards; the script only sets --ids, and
# sets experiment_dir to the experiment holding the config when the folder was copied or moved.
# Env: FLUOROREG_PYTHON (default: Slicer's PythonSlicer, else python3), FLUOROREG_SETTINGS.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"

project="" case="" config="" app="" name="" femur="" predict=(--predict-poses) replay=()
ids=()
need() { if [ $# -lt 2 ] || [ -z "$2" ]; then echo "$1 needs a value" >&2; exit 2; fi; }
while [ $# -gt 0 ]; do
    case "$1" in
        --project) need "$@"; project="$2"; shift ;;
        --case) need "$@"; case="$2"; shift ;;
        --config) need "$@"; config="$2"; shift ;;
        --app) need "$@"; app="$2"; shift ;;
        --name) need "$@"; name="$2"; shift ;;
        --femur) need "$@"; femur="$2"; shift ;;
        --no-predict) predict=() ;;
        --replay) replay=(--replay) ;;
        --ids)
            while [ $# -gt 1 ] && [[ "$2" != --* ]]; do ids+=("$2"); shift; done ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
if [ -z "$project" ]; then echo "--project is required (see --help)" >&2; exit 2; fi
if { [ -z "$config" ] && [ -z "$app" ]; } || { [ -n "$config" ] && [ -n "$app" ]; }; then
    echo "give exactly one of --config and --app (see --help)" >&2; exit 2
fi
project="$(realpath "$project")"
if [ -n "$config" ]; then config="$(realpath "$config")"; fi

if [ -n "$app" ]; then
    if [ -z "$case" ]; then
        case="$("$py" -c 'import sys
from fluororeg.project import load_manifest
cases = [c["case_id"] for c in load_manifest(sys.argv[1]).get("cases", [])]
if len(cases) != 1:
    sys.exit("the project has %d cases (%s); pass --case" % (len(cases), " ".join(cases)))
print(cases[0])' "$project")"
    fi
    if [ -z "$name" ]; then
        # Through the environment: the PythonSlicer launcher prints its usage (to stdout) for an empty argument.
        name="$(APP="$app" CASE="$case" FEMUR="$femur" "$py" -c 'import os
from fluororeg.configs import default_name
print(default_name(os.environ["APP"], os.environ["CASE"], os.environ["FEMUR"] or None))')"
    fi
    config="$project/configs/xreg/$name.json"
    if [ ! -f "$config" ]; then
        create=(config create --project "$project" --case "$case" --app "$app" --name "$name")
        if [ -n "$femur" ]; then create+=(--femur "$femur"); fi
        if [ ${#ids[@]} -gt 0 ]; then create+=(--ids "${ids[@]}"); fi
        pipeline "${create[@]}"
    elif [ ${#ids[@]} -gt 0 ]; then
        pipeline config set --config "$config" --ids "${ids[@]}"
    else
        echo "= using configs/xreg/$name.json as it is"
    fi
fi

# A copied or moved experiment: its configs still name the old experiment_dir.
holder="$(CONFIG="$config" "$py" -c 'import os
from fluororeg.configs import holding_project, is_relocated, load_config
path = os.environ["CONFIG"]
if is_relocated(load_config(path), path):
    print(holding_project(path))')"
if [ -n "$holder" ]; then
    echo "= the config names another experiment_dir; setting it to $holder"
    pipeline config set --config "$config" --experiment-dir "$holder"
fi

rc=0
pipeline register --config "$config" "${predict[@]}" "${replay[@]}" || rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then exit "$rc"; fi
echo "$result" | PROJECT="$project" CONFIG="$config" "$py" -c 'import json, os, sys
from fluororeg.configs import load_config
r = json.load(sys.stdin)
experiment = load_config(os.environ["CONFIG"]).get("experiment_dir") or os.environ["PROJECT"]
counts = {c: 0 for c in ("high", "medium", "low")}
for item in r.get("items") or []:
    counts[item["certainty"]] = counts.get(item["certainty"], 0) + 1
print()
print("Run: " + os.path.join(experiment, r["run_dir"]))
print(", ".join(f"{n} {c} certainty" for c, n in counts.items()))'
if [ "$rc" -eq 3 ]; then echo "Needs attention: check the medium- or low-certainty items in the viewer."; fi

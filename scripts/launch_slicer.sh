#!/usr/bin/env bash
# Opens 3D Slicer with the FluoroReg module, optionally opening an experiment or run folder.
#
#   scripts/launch_slicer.sh [--module NAME] [--headless] [--webserver [PORT]] [FOLDER]
#
#   --module NAME      Slicer module selected at startup (default: FluoroReg)
#   --headless         run under xvfb-run (no window on your display; use with --webserver)
#   --webserver [PORT] start Slicer's WebServer with exec enabled on 127.0.0.1:PORT (default 2017);
#                      only when asked: it lets any local process run Python in Slicer
#   FOLDER             experiment folder, or a run folder (run_manifest.json) inside one; opened in FluoroReg
#                      (a run folder opens its experiment and shows the run)
#
# Env: SLICER (Slicer executable; default: "slicer" from scripts/configure.sh, else Slicer on PATH).
set -euo pipefail

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
headless=0
port=""
folder=""
module=""

while [ $# -gt 0 ]; do
    case "$1" in
        --module)
            if [ $# -lt 2 ]; then echo "--module needs a module name" >&2; exit 2; fi
            module="$2"; shift ;;
        --headless) headless=1 ;;
        --webserver)
            if [ $# -gt 1 ] && [[ "$2" =~ ^[0-9]+$ ]]; then port="$2"; shift; else port=2017; fi ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)
            if [ -n "$folder" ]; then echo "only one folder" >&2; exit 2; fi
            folder="$1" ;;
    esac
    shift
done

source "$repo/scripts/module_dirs.sh"
if [ -n "$folder" ]; then
    if [ ! -d "$folder" ]; then echo "not a folder: $folder" >&2; exit 2; fi
    folder="$(cd "$folder" && pwd)"
fi
module="${module:-FluoroReg}"
source "$repo/scripts/find_slicer.sh"

export FLUOROREG_START_MODULE="$module"
export FLUOROREG_OPEN="$folder"
export FLUOROREG_SLICER_WEBSERVER_PORT="$port"
cmd=("$slicer_bin" --no-splash --additional-module-paths "${module_dirs[@]}"
     --python-script "$repo/scripts/slicer_startup.py")

if [ "$headless" -eq 1 ]; then
    exec xvfb-run -a -s "-screen 0 1920x1200x24" "${cmd[@]}" --disable-settings
fi
exec "${cmd[@]}"

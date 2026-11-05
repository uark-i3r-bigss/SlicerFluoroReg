# Sourced by the scripts that run 3D Slicer. Sets slicer_bin to $SLICER if set, else to "slicer" in the
# settings file written by scripts/configure.sh, else to Slicer found on PATH, and slicer_python to the
# PythonSlicer of that installation; exits with a message otherwise.

settings_file="${FLUOROREG_SETTINGS:-$HOME/.config/fluororeg/settings.json}"
slicer_bin="${SLICER:-}"
if [ -z "$slicer_bin" ] && [ -f "$settings_file" ]; then
    slicer_bin="$("${FLUOROREG_PYTHON:-python3}" -c \
        'import json, sys; print(json.load(open(sys.argv[1])).get("slicer", ""))' "$settings_file" 2>/dev/null || true)"
fi
if [ -z "$slicer_bin" ] && ! slicer_bin="$(command -v Slicer)"; then
    echo "Slicer not found: run scripts/configure.sh --slicer <Slicer install dir>/Slicer (or set SLICER)" >&2
    exit 2
fi
if [ ! -f "$slicer_bin" ] || [ ! -x "$slicer_bin" ]; then
    echo "Slicer path $slicer_bin is not an executable file: rerun scripts/configure.sh --slicer <path> (or set SLICER)" >&2
    exit 2
fi
slicer_python="$(dirname "$(readlink -f "$slicer_bin")")/bin/PythonSlicer"

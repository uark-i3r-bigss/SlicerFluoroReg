# Sourced by the quick-start scripts. Sets repo and py, puts the pipeline package on PYTHONPATH and defines:
#   pipeline ARGS...  run `$py -m fluororeg ARGS...` with live output; the last result event goes to
#                     $result (JSON, "null" without one); returns the command's exit status
#   json_get EXPR     print a Python expression of r (the JSON read from stdin), e.g. json_get 'r["case"]'
#   usage             print the header comment of the calling script
# py: $FLUOROREG_PYTHON, else the PythonSlicer next to the settings file's "slicer" (it has pydicom, numpy and
# Pillow for image X-rays), else python3.

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
settings_file="${FLUOROREG_SETTINGS:-$HOME/.config/fluororeg/settings.json}"
py="${FLUOROREG_PYTHON:-}"
if [ -z "$py" ] && [ -f "$settings_file" ]; then
    slicer_exe="$(sed -n 's/^[[:space:]]*"slicer"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$settings_file")"
    if [ -n "$slicer_exe" ] && [ -x "$(dirname "$slicer_exe")/bin/PythonSlicer" ]; then
        py="$(dirname "$slicer_exe")/bin/PythonSlicer"
    fi
fi
py="${py:-python3}"
export PYTHONPATH="$repo/FluoroReg${PYTHONPATH:+:$PYTHONPATH}"
result="null"

usage() { sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

pipeline() {
    local out rc=0
    out="$(mktemp)"
    echo "+ $(basename "$py") -m fluororeg $*"
    "$py" -m fluororeg "$@" | tee "$out" || rc=$?
    result="$("$py" - "$out" <<'EOF'
import json, sys
last = None
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if line.startswith("FLUOROREG "):
        try:
            event = json.loads(line[len("FLUOROREG "):])
        except ValueError:
            continue
        if event.get("event") == "result":
            last = event
print(json.dumps(last))
EOF
)"
    rm -f "$out"
    return "$rc"
}

json_get() {
    "$py" -c 'import json, sys
r = json.load(sys.stdin)
value = eval(sys.argv[1])
print(value if isinstance(value, str) else json.dumps(value))' "$1"
}

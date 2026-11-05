#!/usr/bin/env bash
# Runs, in order: the local unit tests when tests/ exists (a python3 with numpy, and Slicer's Python 3.9),
# the contract check of every run directory, and the headless Slicer self-test of FluoroReg (run directories
# passed in $FLUOROREG_TEST_RUNS). The module directory is on PYTHONPATH and a Slicer module path.
#
#   scripts/run_tests.sh [RUN_DIR ...]
#
# Run directories come from the arguments or, when none are given, from $FLUOROREG_TEST_RUNS
# (separated by ':' or whitespace). Without runs only the real-run contract check is skipped;
# Slicer's fixture self-test still runs and must pass.
# Env: PYTHON (interpreter for the library; default python3 when it has numpy, else Slicer's),
#      SLICER (Slicer executable; default: "slicer" from scripts/configure.sh, else Slicer on PATH).
# Exit status: 0 when executed steps pass (see explicit skips), 1 otherwise.
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/scripts/find_slicer.sh"
source "$repo/scripts/module_dirs.sh"

runs=("$@")
if [ ${#runs[@]} -eq 0 ] && [ -n "${FLUOROREG_TEST_RUNS:-}" ]; then
    read -r -a runs <<< "$(echo "$FLUOROREG_TEST_RUNS" | tr ':' ' ')"
fi

python="${PYTHON:-}"
if [ -z "$python" ]; then
    if python3 -c "import numpy" 2>/dev/null; then python=python3; else python="$slicer_python"; fi
fi

export PYTHONPATH="$module_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
failed=()
passed=0
skipped=()

step() {
    local name="$1"; shift
    echo "=== $name"
    if "$@"; then
        echo "=== $name: PASS"
        passed=$((passed + 1))
    else
        echo "=== $name: FAIL"
        failed+=("$name")
    fi
}

if [ -d "$repo/tests" ]; then
    step "unit tests ($("$python" -c 'import sys; print(sys.executable, sys.version.split()[0])'))" \
        "$python" -m unittest discover -s "$repo/tests"
    if [ -x "$slicer_python" ] && [ "$python" != "$slicer_python" ]; then
        step "unit tests (Slicer Python)" "$slicer_python" -m unittest discover -s "$repo/tests"
    fi
else
    skipped+=("local unit tests (no tests/ folder)")
fi

if [ ${#runs[@]} -gt 0 ]; then
    step "contract check" "$python" -m fluororeg_runs check "${runs[@]}"
else
    skipped+=("real-run contract check (no run directories)")
fi
selftest=("$slicer_bin" --no-splash --testing --disable-settings --no-main-window
          --additional-module-paths "${module_dirs[@]}" --python-script "$repo/scripts/slicer_selftest.py")
if command -v xvfb-run >/dev/null; then
    selftest=(xvfb-run -a -s "-screen 0 1280x1024x24" "${selftest[@]}")
fi
step "Slicer fixture self-test and supplied runs (${module_names[*]})" \
    env FLUOROREG_TEST_RUNS="${runs[*]}" "${selftest[@]}"

for reason in "${skipped[@]}"; do
    echo "SKIPPED: $reason"
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
if [ "$passed" -eq 0 ]; then
    echo "FAILED: no checks executed"
    exit 1
fi
echo "PASSED: $passed executed step(s); SKIPPED: ${#skipped[@]} step(s)"

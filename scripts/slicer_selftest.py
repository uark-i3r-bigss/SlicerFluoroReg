"""Runs, inside Slicer, the self-test of every module of this repository that has one (modules: the
add_subdirectory entries of the top-level CMakeLists.txt) and exits with 0 (all pass) or 1 (a module is not
loaded, has no test at all, or a test fails).

Use, with the run directories in $FLUOROREG_TEST_RUNS (read by the FluoroReg self-test):
  the repository's modules:  Slicer --no-splash --testing --disable-settings --no-main-window
                               --additional-module-paths <repo>/<module> [...] --python-script scripts/slicer_selftest.py
  the installed extension:   Slicer --no-splash --no-main-window --python-script scripts/slicer_selftest.py
                               (no --testing/--disable-settings: the extension is loaded from the settings)
"""
import importlib.util
import traceback
from pathlib import Path

import slicer

_scripts = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("package_extension", str(_scripts / "package_extension.py"))
package_extension = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(package_extension)

code = 0
tested = 0
for name in package_extension.module_names(_scripts.parent):
    try:
        module = getattr(slicer.modules, name.lower(), None)
        if module is None:
            print(f"FLUOROREG_SELFTEST_MODULE {name} not loaded", flush=True)
            code = 1
            continue
        print(f"FLUOROREG_SELFTEST_MODULE {name} {module.path}", flush=True)
        if name not in slicer.selfTests:
            print(f"FLUOROREG_SELFTEST_MODULE {name} has no self-test", flush=True)
            continue
        tested += 1
        slicer.selfTests[name]()
        print(f"FLUOROREG_SELFTEST_MODULE {name} PASS", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"FLUOROREG_SELFTEST_MODULE {name} FAIL", flush=True)
        code = 1
if tested == 0:
    print("FLUOROREG_SELFTEST no module has a self-test", flush=True)
    code = 1
print(f"FLUOROREG_SELFTEST_EXIT {code}", flush=True)
slicer.util.exit(code)

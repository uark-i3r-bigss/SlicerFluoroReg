#-----------------------------------------------------------------------------
set(MODULE_NAME FluoroReg)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  fluororeg/__init__.py
  fluororeg/__main__.py
  fluororeg/assess.py
  fluororeg/backends.py
  fluororeg/cli.py
  fluororeg/configs.py
  fluororeg/ct.py
  fluororeg/landmarks.py
  fluororeg/pose.py
  fluororeg/project.py
  fluororeg/register.py
  fluororeg/report.py
  fluororeg/runner.py
  fluororeg/settings.py
  fluororeg/status.py
  fluororeg/xrays.py
  fluororeg_viewer/__init__.py
  fluororeg_viewer/checks.py
  fluororeg_viewer/scene.py
  fluororeg_runs/__init__.py
  fluororeg_runs/__main__.py
  fluororeg_runs/camera.py
  fluororeg_runs/contract.py
  fluororeg_runs/fcsv.py
  fluororeg_runs/imaging.py
  fluororeg_runs/landmarks.py
  fluororeg_runs/manifest.py
  fluororeg_runs/mha.py
  fluororeg_runs/scene.py
  fluororeg_runs/transforms.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  fluororeg/templates/multi-target-multiview.json
  fluororeg/templates/multi-target.json
  fluororeg/templates/single-target-multiview.json
  fluororeg/templates/single-target.json
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

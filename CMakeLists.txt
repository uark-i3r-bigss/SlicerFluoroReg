cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerFluoroReg)

#-----------------------------------------------------------------------------
# Extension meta-information (also read by scripts/package_extension.py)
set(EXTENSION_HOMEPAGE "https://github.com/uark-i3r-bigss/SlicerFluoroReg")
set(EXTENSION_VERSION "2.0.0")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "BIGSS lab")
set(EXTENSION_DESCRIPTION "SlicerFluoroReg: automatic 2D/3D registration of C-arm radiographs to CT. One module, FluoroReg, runs a pipeline from CT and X-ray DICOMs to registered bone poses (learned initial pose, xReg single-view, multiview and pelvis-then-femur registration, certainty grade) and shows the runs, with a contract check for xReg runs (run_manifest.json schema 4)")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/uark-i3r-bigss/SlicerFluoroReg/main/FluoroReg/Resources/Icons/FluoroReg.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(FluoroReg)

# Legal notices and operator documentation must accompany CPack installations too.
set(FLUOROREG_DOCUMENTATION_DIR
  "share/Slicer-${Slicer_VERSION_MAJOR}.${Slicer_VERSION_MINOR}/${PROJECT_NAME}")
install(FILES
  LICENSE
  THIRD_PARTY_NOTICES
  README.md
  DESTINATION "${FLUOROREG_DOCUMENTATION_DIR}"
  COMPONENT Runtime)
install(FILES
  docs/setup.md
  docs/pipeline.md
  docs/developer.md
  DESTINATION "${FLUOROREG_DOCUMENTATION_DIR}/docs"
  COMPONENT Runtime)

# Share the tracked vendor allowlist and payload inventory with the local archive builder.
# Configure only reads source files; backend installation is always an explicit operator job.
find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
set(FLUOROREG_VENDOR_INSTALL "${CMAKE_CURRENT_BINARY_DIR}/FluoroRegVendorInstall.cmake")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/package_extension.py"
    --slicer-version "${Slicer_VERSION_MAJOR}.${Slicer_VERSION_MINOR}"
    --cmake-support "${FLUOROREG_VENDOR_INSTALL}"
  RESULT_VARIABLE FLUOROREG_VENDOR_RESULT
  ERROR_VARIABLE FLUOROREG_VENDOR_ERROR)
if(NOT FLUOROREG_VENDOR_RESULT EQUAL 0)
  message(FATAL_ERROR "Cannot prepare curated backend package: ${FLUOROREG_VENDOR_ERROR}")
endif()
include("${FLUOROREG_VENDOR_INSTALL}")

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

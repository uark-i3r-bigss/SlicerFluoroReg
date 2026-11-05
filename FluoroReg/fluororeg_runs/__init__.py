"""Pure-Python (numpy only) reader and contract check for xreg run directories.

No Slicer, VTK, Qt or h5py imports: usable from plain Python 3.9+ and from 3D Slicer.
"""
from .camera import RawCamera
from .contract import ContractReport, contract_check
from .fcsv import read_fcsv, read_fcsv_text
from .landmarks import invert_registration_names, landmark_residuals_px, registration_points_2d
from .manifest import MANIFEST_NAME, POSE_STAGES, ManifestError, XregRun
from .mha import read_mha, read_mha_header
from .scene import SceneMatrices, scene_matrices
from .transforms import (RAS_FROM_LPS, primary_from_target, ras_from_camworld, slicer_to_parent_of_itk_file)

__all__ = [
    "RawCamera", "ContractReport", "contract_check", "read_fcsv", "read_fcsv_text",
    "invert_registration_names", "landmark_residuals_px", "registration_points_2d", "MANIFEST_NAME",
    "POSE_STAGES", "ManifestError", "XregRun", "read_mha", "read_mha_header", "SceneMatrices",
    "scene_matrices", "RAS_FROM_LPS", "primary_from_target", "ras_from_camworld",
    "slicer_to_parent_of_itk_file",
]

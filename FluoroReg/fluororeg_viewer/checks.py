"""Run checks of the self-test: a run loaded into Slicer matches its manifest (inputs, transform files, images,
pose stages, landmark reprojection) and a CT already in the scene is reused."""
import os

import numpy as np
import vtk

import slicer

from fluororeg_runs import XregRun
from fluororeg_runs.mha import read_mha
from fluororeg_runs.scene import default_stages
from fluororeg_runs.transforms import (RAS_FROM_LPS, apply, inv, lps_to_ras, primary_from_target,
                                       slicer_to_parent_of_itk_file)

from .scene import RunViewer

TEST_RUNS_ENV = "FLUOROREG_TEST_RUNS"


def test_run_dirs():
    """Run directories of $FLUOROREG_TEST_RUNS (separated by os.pathsep or white space)."""
    raw = os.environ.get(TEST_RUNS_ENV, "")
    return [p for p in raw.replace(os.pathsep, "\n").split() if p]


class RunChecks:
    """Mixin of a ScriptedLoadableModuleTest: checks the Slicer scene of a run against its manifest."""

    def check_run(self, path, residual_tol_px=0.5, matrix_tol=1e-4):
        viewer = RunViewer()
        scene = viewer.load_run(path, load_ct=True, build_models=True, volume_rendering=True)
        run = scene.run
        self.delayDisplay(f"loaded {run.name} ({run.app})")
        report = {"run": run.name, "app": run.app, "experiment_dir": str(run.experiment_dir)}
        self._check_inputs(scene)
        report["transform_files"], report["max_file_matrix_diff"] = self._check_transform_files(run, matrix_tol)
        report["images"] = self._check_images(scene)
        report["stage_switches"] = self._check_stage_switching(scene, matrix_tol)
        n, worst = self._check_reprojection(scene, residual_tol_px)
        report["residuals_compared"], report["max_residual_diff_px"] = n, worst
        viewer.clear()
        print(f"FLUOROREG_RUN_CHECK {report}", flush=True)
        return report

    def check_ct_reuse(self, path):
        """A CT already in the scene (same file) is reused, not loaded again, and survives clearing the run
        with its previous parent transform."""
        ct_path = XregRun.load(path).input_path("ct")
        ct = slicer.util.loadVolume(str(ct_path))
        parent = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "user transform")
        ct.SetAndObserveTransformNodeID(parent.GetID())
        viewer = RunViewer()
        scene = viewer.load_run(path, load_ct=True, build_models=False, volume_rendering=True)
        if scene.ct is None or scene.ct.GetID() != ct.GetID():
            raise AssertionError(f"{scene.run.name}: the scene CT was not reused")
        copies = [n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                  if n.GetStorageNode() is not None and n.GetStorageNode().GetFileName()
                  and os.path.realpath(n.GetStorageNode().GetFileName()) == os.path.realpath(str(ct_path))]
        if len(copies) != 1:
            raise AssertionError(f"{scene.run.name}: {len(copies)} volumes hold {ct_path}")
        viewer.clear()
        if ct.GetScene() is None or ct.GetTransformNodeID() != parent.GetID():
            raise AssertionError(f"{scene.run.name}: clearing the run removed or moved the reused CT")
        if slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(ct) is not None:
            raise AssertionError(f"{scene.run.name}: the run's volume rendering stayed on the reused CT")

    def _check_inputs(self, scene):
        """The CT was read from <experiment folder>/<inputs.ct.path>, the folder derived from run.dir."""
        want = scene.run.input_path("ct")
        got = scene.ct.GetStorageNode().GetFileName() if scene.ct is not None else None
        if got is None or os.path.normpath(got) != os.path.normpath(str(want)):
            raise AssertionError(f"{scene.run.name}: CT loaded from {got}, expected {want}")

    def _check_transform_files(self, run, tol):
        count, worst = 0, 0.0
        for t in run.transforms:
            if not str(t.get("file") or "").endswith(".h5"):
                continue
            node = slicer.util.loadTransform(str(run.run_file(t["file"])))
            if node is None:
                raise AssertionError(f"{run.name}: Slicer could not load {t['file']}")
            got = slicer.util.arrayFromTransformMatrix(node)
            want = slicer_to_parent_of_itk_file(run.matrix(t))
            diff = float(np.max(np.abs(got - want)))
            slicer.mrmlScene.RemoveNode(node)
            worst = max(worst, diff)
            count += 1
            if diff > tol:
                raise AssertionError(f"{run.name}: {t['file']} loads as to-parent\n{got}\nexpected C inv(M) C\n{want}")
        if count == 0:
            raise AssertionError(f"{run.name}: no .h5 transform files in the manifest")
        return count, worst

    def _check_images(self, scene):
        """Image nodes have the camera's dimensions and spacing and hold the files' pixels as stored
        (row 0 first); the detector plane's texture coordinates land on the X-ray node's pixel edges."""
        count = 0
        for view_id, vn in scene.views.items():
            run = scene.run
            cam = run.camera(view_id)
            want = (cam.cols, cam.rows, 1)
            for node in [vn.xray, vn.composite] + list(vn.drrs.values()):
                dims = node.GetImageData().GetDimensions()
                if tuple(dims) != want:
                    raise AssertionError(f"view {view_id}: {node.GetName()} has dimensions {dims}, camera {want}")
                spacing = node.GetSpacing()
                if not np.allclose(spacing[:2], (cam.col_spacing_mm, cam.row_spacing_mm), atol=1e-6):
                    raise AssertionError(f"view {view_id}: {node.GetName()} spacing {spacing}")
                count += 1
            files = [(vn.xray, run.image_file(view_id, "xray_raw"))]
            files += [(node, run.drr_file(view_id, key)) for key, node in vn.drrs.items()]
            for node, path in files:
                if not np.array_equal(slicer.util.arrayFromVolume(node)[0], read_mha(str(path))[1]):
                    raise AssertionError(f"view {view_id}: {node.GetName()} pixels differ from {path}")
            if len(vn.drrs) != len([k for k in run.target_keys if run.drr_file(view_id, k)]):
                raise AssertionError(f"view {view_id}: DRR nodes missing")

            m = vtk.vtkMatrix4x4()
            vn.xray.GetIJKToRASMatrix(m)
            local_to_pix = inv(slicer.util.arrayFromVTKMatrix(m))
            if vn.detector.GetTransformNodeID() != vn.xray.GetTransformNodeID():
                raise AssertionError(f"view {view_id}: detector plane and X-ray have different parents")
            poly = vn.detector.GetPolyData()
            tcoords = poly.GetPointData().GetTCoords()
            for i in range(poly.GetNumberOfPoints()):
                s, t = tcoords.GetTuple2(i)
                pix = apply(local_to_pix, poly.GetPoint(i))
                expected = [s * cam.cols - 0.5, t * cam.rows - 0.5, 0.0]
                if np.max(np.abs(pix - expected)) > 1e-3:
                    raise AssertionError(f"view {view_id}: texture coordinate ({s}, {t}) is at pixel {pix}, "
                                         f"expected {expected}")
        return count

    def _check_stage_switching(self, scene, tol):
        """Camera and anatomy nodes, read back from Slicer, equal the manifest pose of every stage, in the
        CT-centred frame and in the camera-centred frame (where the selected camera is the identity)."""
        run, primary, count = scene.run, scene.primary, 0
        for camera_centred in (False, True):
            scene.set_camera_centred(camera_centred)
            for view_id in run.view_ids:
                if not run.pose_stages(view_id, primary):
                    continue
                scene.set_view(view_id)
                for key in run.target_keys:
                    stages = run.pose_stages(view_id, key)
                    for stage in stages:
                        scene.set_stage(key, stage)
                        p = run.pose_matrix(view_id, primary, scene.stages[primary])
                        want_carm = RAS_FROM_LPS @ p
                        want_anat = np.eye(4) if key == primary else lps_to_ras(
                            primary_from_target(p, run.pose_matrix(view_id, key, stage)))
                        if camera_centred:
                            want_carm, want_anat = inv(want_carm) @ want_carm, inv(want_carm) @ want_anat
                        carm = slicer.util.arrayFromTransformMatrix(scene.views[view_id].carm, toWorld=True)
                        anat = slicer.util.arrayFromTransformMatrix(scene.anatomy[key], toWorld=True)
                        if np.max(np.abs(carm - want_carm)) > tol or np.max(np.abs(anat - want_anat)) > tol:
                            raise AssertionError(f"view {view_id} {key} {stage} (camera_centred={camera_centred}): "
                                                 f"node matrices differ from the manifest")
                        count += 1
                    if stages:
                        scene.set_stage(key, default_stages(run, view_id)[key])
        scene.set_camera_centred(False)
        return count

    def _check_reprojection(self, scene, tol):
        """xreg's landmark residuals, recomputed only from what Slicer holds: 3D landmark and 2D
        landmark world positions, the C-arm/anatomy transforms, the X-ray image geometry and the
        frustum's source point. The 2D landmarks must also sit on their annotated pixels of the X-ray
        image node (catches an image placed mirrored or shifted on the detector)."""
        run, count, worst = scene.run, 0, 0.0
        for view in run.views:
            view_id = view["id"]
            targets = {k: e for k, e in (view.get("targets") or {}).items()
                       if (e.get("landmark_residual_px") or {}).get("per_landmark") and e.get("final")}
            if not targets:
                continue
            scene.set_view(view_id)
            for key in run.target_keys:
                if "final" in run.pose_stages(view_id, key):
                    scene.set_stage(key, "final")
            for camera_centred in (False, True):
                scene.set_camera_centred(camera_centred)
                vn = scene.views[view_id]
                pix_to_world = scene.pixel_to_world(view_id)
                world_to_pix = inv(pix_to_world)
                frustum_world = slicer.util.arrayFromTransformMatrix(vn.frustum.GetParentTransformNode(), toWorld=True)
                source = apply(frustum_world, vn.frustum.GetPolyData().GetPoint(0))
                lm2d = self._world_points(vn.landmarks_2d)
                for name, uv in run.points_2d(view_id).items():
                    on_image = apply(world_to_pix, lm2d[name])[:2]
                    if np.max(np.abs(on_image - uv)) > 1e-3:
                        raise AssertionError(f"view {view_id}: 2D landmark {name} is at pixel {on_image} of the "
                                             f"X-ray node, annotated at {uv}")
                for key, entry in targets.items():
                    lm3d = self._world_points(scene.landmarks_3d[key])
                    for name, reported in entry["landmark_residual_px"]["per_landmark"].items():
                        if name not in lm3d or name not in lm2d:
                            raise AssertionError(f"view {view_id} {key}: landmark {name} missing from the scene")
                        uv3 = self._ray_to_pixel(source, lm3d[name], pix_to_world)
                        uvw2 = apply(world_to_pix, lm2d[name])
                        if abs(uvw2[2]) > 1e-3:
                            raise AssertionError(f"view {view_id}: 2D landmark {name} is off the detector plane")
                        ours = float(np.hypot(*(uv3 - uvw2[:2])))
                        diff = abs(ours - float(reported))
                        worst = max(worst, diff)
                        count += 1
                        if diff > tol:
                            raise AssertionError(
                                f"view {view_id} {key} {name} (camera_centred={camera_centred}): Slicer residual "
                                f"{ours:.3f} px, xreg {float(reported):.3f} px")
            scene.set_camera_centred(False)
        return count, worst

    @staticmethod
    def _world_points(markups):
        out = {}
        p = [0.0, 0.0, 0.0]
        for i in range(markups.GetNumberOfControlPoints()):
            markups.GetNthControlPointPositionWorld(i, p)
            out[markups.GetNthControlPointDescription(i)] = np.array(p)
        return out

    @staticmethod
    def _ray_to_pixel(source, point, pix_to_world):
        """(u, v) where the ray source -> point meets the image plane of pix_to_world."""
        o, a, b = pix_to_world[:3, 3], pix_to_world[:3, 0], pix_to_world[:3, 1]
        d = np.asarray(point) - np.asarray(source)
        s, u, v = np.linalg.solve(np.column_stack([d, -a, -b]), o - np.asarray(source))
        return np.array([u, v])

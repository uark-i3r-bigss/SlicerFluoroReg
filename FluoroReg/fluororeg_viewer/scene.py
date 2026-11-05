"""Scene of an xreg registration run directory (run_manifest.json schema 4) in 3D Slicer.

Scene: the CT (optional volume rendering), models of the target segmentation labels, the 3D
landmarks, and per X-ray a C-arm transform RAS_from_CAMWORLD = RAS_from_LPS @ CT_from_CAMWORLD
holding the frustum, the raw X-ray (with DRR overlays) on the detector plane of the raw camera and
the 2D landmarks on the detector. All geometry comes from explicit manifest matrices; the math
lives in the Slicer-free library fluororeg_runs. Inputs (CT, segmentation, 3D landmarks) are read
from the experiment folder derived from the manifest's run.dir, so a copied experiment folder opens
anywhere.
"""
import logging
import os

import numpy as np
import vtk

import slicer

from fluororeg_runs import XregRun
from fluororeg_runs.imaging import composite_rgb
from fluororeg_runs.mha import read_mha
from fluororeg_runs.scene import default_stages, scene_matrices
from fluororeg_runs.transforms import RAS_FROM_LPS, apply

MODEL_COLORS = [(0.93, 0.86, 0.70), (0.55, 0.75, 1.00), (0.60, 1.00, 0.60), (1.00, 0.70, 0.55)]
DRR_COLORS = [(0.10, 1.00, 0.10), (1.00, 0.15, 1.00), (0.10, 0.90, 1.00), (1.00, 0.85, 0.10)]
LANDMARK_3D_COLORS = [(1.00, 0.85, 0.00), (0.20, 0.60, 1.00), (0.20, 1.00, 0.40), (1.00, 0.50, 0.00)]
LANDMARK_2D_COLOR = (1.00, 0.20, 0.10)
FRUSTUM_COLOR = (0.55, 0.60, 0.70)
FRUSTUM_SELECTED_COLOR = (1.00, 0.90, 0.10)


#
# Scene of one loaded run
#
def _short(name):
    return name.split("::", 1)[-1]


def _set_matrix(transform_node, m):
    transform_node.SetMatrixTransformToParent(slicer.util.vtkMatrixFromArray(np.asarray(m, dtype=float)))


def _new_transform(name):
    return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)


def _markups(name, points, labels, descriptions, color, glyph_mm, parent_transform):
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
    node.CreateDefaultDisplayNodes()
    for p, label, desc in zip(points, labels, descriptions):
        i = node.AddControlPoint([float(p[0]), float(p[1]), float(p[2])], label)
        node.SetNthControlPointDescription(i, desc)
    node.SetLocked(True)
    d = node.GetDisplayNode()
    d.SetColor(*color)
    d.SetSelectedColor(*color)
    d.SetActiveColor(*color)
    d.SetUseGlyphScale(False)
    d.SetGlyphSize(glyph_mm)
    d.SetTextScale(2.5)
    d.SetPointLabelsVisibility(True)
    node.SetAndObserveTransformNodeID(parent_transform.GetID())
    return node


def _scene_volume_with_file(path):
    """A scene scalar volume (not a label map) whose storage file is ``path`` (realpath), else None."""
    target = os.path.realpath(str(path))
    for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        storage = node.GetStorageNode()
        name = storage.GetFileName() if storage is not None else None
        if name and not node.IsA("vtkMRMLLabelMapVolumeNode") and os.path.realpath(name) == target:
            return node
    return None


def _label_surface(labels_kji, ijk_to_ras, label, sigma_mm=2.0):
    """Smoothed surface (RAS) of one label of a label volume array, or None when the label is absent.

    The binary mask is Gaussian-blurred (sigma_mm) before contouring at 0.5, which removes the voxel
    staircase of labels drawn on a coarser grid than the CT (steps of a few mm)."""
    from vtk.util.numpy_support import numpy_to_vtk

    mask = labels_kji == label
    if not mask.any():
        return None
    pad = 4
    ranges = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        idx = np.flatnonzero(mask.any(axis=other))
        ranges.append((int(idx[0]), int(idx[-1]) + 1))
    (k0, k1), (j0, j1), (i0, i1) = ranges
    sub = np.pad(mask[k0:k1, j0:j1, i0:i1], pad).astype(np.float32)
    img = vtk.vtkImageData()
    img.SetDimensions(sub.shape[2], sub.shape[1], sub.shape[0])
    img.SetOrigin(i0 - pad, j0 - pad, k0 - pad)
    img.GetPointData().SetScalars(numpy_to_vtk(sub.ravel(), deep=True))

    spacing = np.linalg.norm(np.asarray(ijk_to_ras)[:3, :3], axis=0)
    blur = vtk.vtkImageGaussianSmooth()
    blur.SetInputData(img)
    blur.SetStandardDeviations(*(sigma_mm / spacing))
    blur.SetRadiusFactors(2.0, 2.0, 2.0)
    surface = vtk.vtkFlyingEdges3D()
    surface.SetInputConnection(blur.GetOutputPort())
    surface.SetValue(0, 0.5)
    surface.ComputeNormalsOff()
    surface.ComputeScalarsOff()
    xform = vtk.vtkTransform()
    xform.SetMatrix(slicer.util.vtkMatrixFromArray(ijk_to_ras))
    to_ras = vtk.vtkTransformPolyDataFilter()
    to_ras.SetInputConnection(surface.GetOutputPort())
    to_ras.SetTransform(xform)
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(to_ras.GetOutputPort())
    smooth.SetNumberOfIterations(15)
    smooth.SetPassBand(0.01)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(normals.GetOutput())
    return out


def _frustum_polydata(points_camworld):
    """Lines from the source (row 0) to the four detector corners and around the detector."""
    pts = vtk.vtkPoints()
    for p in points_camworld:
        pts.InsertNextPoint(*[float(x) for x in p])
    lines = vtk.vtkCellArray()
    for i in range(1, 5):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(i)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(i)
        lines.InsertCellPoint(1 + i % 4)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
    return poly


class ViewNodes:
    """Nodes of one X-ray view; everything but the C-arm transform lives in CAMWORLD[view]."""

    def __init__(self):
        self.carm = None
        self.frustum = None
        self.detector = None      # textured plane model showing the composite
        self.xray = None          # scalar volume, raw X-ray on the detector plane
        self.drrs = {}            # target key -> scalar volume
        self.composite = None     # RGB vector volume: X-ray with DRR overlays (texture of detector)
        self.landmarks_2d = None
        self.xray_pixels = None
        self.drr_pixels = {}


class RunScene:
    """Slicer nodes of one loaded run plus the display state; every method is callable from scripts."""

    def __init__(self, run):
        self.run = run
        self.primary = run.primary_key
        self.nodes = []
        self.folder = None
        self.anatomy = {}         # target key -> transform node
        self.models = {}          # target key -> model node
        self.landmarks_3d = {}    # target key (or "other") -> markups node
        self.ct = None
        self.reused_ct = None     # (scene CT node not owned by the run, its previous transform node id)
        self.views = {}           # view id -> ViewNodes
        self.view_id = None
        self.stages = {}
        self.camera_centred = False
        self.stages_all_views = False
        self.show_all_views = True
        self.drr_overlay = True
        self.drr_opacity = 0.5
        self.models_visible = True
        self.landmarks_visible = True
        self.scene_state = None

    # ------------------------------------------------------------------ bookkeeping
    def _own(self, node, subfolder=None):
        self.nodes.append(node)
        sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        item = sh.GetItemByDataNode(node)
        if item and self.folder is not None:
            sh.SetItemParent(item, subfolder if subfolder is not None else self.folder)
        return node

    def remove(self):
        for node in reversed(self.nodes):
            if node.GetScene() is None:
                continue
            if node.IsA("vtkMRMLDisplayableNode"):
                for i in range(node.GetNumberOfDisplayNodes() - 1, -1, -1):
                    d = node.GetNthDisplayNode(i)
                    if d is not None and d.GetScene() is not None:
                        slicer.mrmlScene.RemoveNode(d)
            if node.IsA("vtkMRMLStorableNode") and node.GetStorageNode() is not None:
                slicer.mrmlScene.RemoveNode(node.GetStorageNode())
            slicer.mrmlScene.RemoveNode(node)
        if self.reused_ct is not None:
            node, transform_id = self.reused_ct
            if node.GetScene() is not None:
                node.SetAndObserveTransformNodeID(transform_id)
        if self.folder is not None:
            sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            sh.RemoveItem(self.folder)
        self.nodes, self.views, self.anatomy, self.models, self.landmarks_3d = [], {}, {}, {}, {}
        self.ct, self.reused_ct = None, None

    # ------------------------------------------------------------------ building
    def build(self, load_ct=True, build_models=True, volume_rendering=False):
        run = self.run
        sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        self.folder = sh.CreateFolderItem(sh.GetSceneItemID(), f"xreg run {run.name}")

        for i, key in enumerate(run.target_keys):
            self.anatomy[key] = self._own(_new_transform(f"CT[{self.primary}]_from_CT[{key}] (RAS)"))

        if load_ct:
            ct_path = run.input_path("ct")
            reused = _scene_volume_with_file(ct_path)  # the CT already in the scene: no second copy
            if reused is not None:
                self.ct = reused
                self.reused_ct = (reused, reused.GetTransformNodeID())
            else:
                self.ct = self._own(slicer.util.loadVolume(str(ct_path)))
            self.ct.SetAndObserveTransformNodeID(self.anatomy[self.primary].GetID())
            if volume_rendering:
                self.set_volume_rendering(True)

        if build_models:
            self._build_models()
        self._build_landmarks_3d()
        for view_id in run.view_ids:
            self._build_view(view_id)

        registered = [v["id"] for v in run.views if run.pose(v["id"], self.primary, "final") is not None]
        first = registered[0] if registered else next(
            (v for v in run.view_ids if run.pose_stages(v, self.primary)), None)
        if first is None:
            raise ValueError(f"{run.run_dir}: no view has a pose of the primary target {self.primary}")
        self.set_view(first)

    def _build_models(self):
        run = self.run
        seg_path = run.input_path("segmentation")
        if seg_path is None:
            logging.warning("run has no segmentation input; no models")
            return
        labels = slicer.util.loadLabelVolume(str(seg_path), {"show": False})
        try:
            arr = slicer.util.arrayFromVolume(labels)
            m = vtk.vtkMatrix4x4()
            labels.GetIJKToRASMatrix(m)
            ijk_to_ras = slicer.util.arrayFromVTKMatrix(m)
            for i, t in enumerate(run.targets):
                poly = _label_surface(arr, ijk_to_ras, int(t["segmentation_label"]))
                if poly is None:
                    logging.warning(f"label {t['segmentation_label']} of target {t['key']} is empty")
                    continue
                model = slicer.modules.models.logic().AddModel(poly)
                model.SetName(f"{t.get('display_name', t['key'])} (label {t['segmentation_label']})")
                d = model.GetDisplayNode()
                d.SetColor(*MODEL_COLORS[i % len(MODEL_COLORS)])
                d.SetOpacity(0.6)
                d.SetScalarVisibility(False)
                d.SetVisibility2D(False)
                model.SetAndObserveTransformNodeID(self.anatomy[t["key"]].GetID())
                self.models[t["key"]] = self._own(model)
        finally:
            slicer.mrmlScene.RemoveNode(labels)

    def _build_landmarks_3d(self):
        run = self.run
        pts = run.landmarks_3d()  # LPS
        groups = {key: [] for key in run.target_keys}
        groups["other"] = []
        for name in pts:
            groups[run.target_of_landmark(name) or "other"].append(name)
        for i, key in enumerate(run.target_keys + ["other"]):
            names = groups[key]
            if not names:
                continue
            ras = [RAS_FROM_LPS[:3, :3] @ pts[n] for n in names]
            color = LANDMARK_3D_COLORS[i % len(LANDMARK_3D_COLORS)] if key != "other" else (0.7, 0.7, 0.7)
            parent = self.anatomy[key if key != "other" else self.primary]
            node = _markups(f"3D landmarks {key}", ras, [_short(n) for n in names], names, color, 4.0, parent)
            if key == "other":
                node.GetDisplayNode().SetVisibility(False)
            self.landmarks_3d[key] = self._own(node)

    def _build_view(self, view_id):
        run = self.run
        sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        sub = sh.CreateFolderItem(self.folder, f"view {view_id}")
        cam = run.camera(view_id)
        vn = ViewNodes()
        vn.view_id = view_id
        vn.carm = self._own(_new_transform(f"RAS_from_CAMWORLD[{view_id}]"), sub)

        frustum = slicer.modules.models.logic().AddModel(_frustum_polydata(cam.frustum_camworld()))
        frustum.SetName(f"frustum {view_id}")
        frustum.GetDisplayNode().SetColor(*FRUSTUM_COLOR)
        frustum.GetDisplayNode().SetLineWidth(2.0)
        frustum.GetDisplayNode().SetVisibility2D(False)
        frustum.SetAndObserveTransformNodeID(vn.carm.GetID())
        vn.frustum = self._own(frustum, sub)

        ijk_to_camworld = cam.camworld_from_pixel()
        xray_path = run.image_file(view_id, "xray_raw")
        _, vn.xray_pixels = read_mha(str(xray_path))
        vn.xray = self._own(slicer.util.addVolumeFromArray(
            vn.xray_pixels[None].astype(np.float32), ijk_to_camworld, f"X-ray {view_id}"), sub)
        vn.xray.SetAndObserveTransformNodeID(vn.carm.GetID())
        vn.xray.GetDisplayNode().AutoWindowLevelOn()
        for key in run.target_keys:
            drr_path = run.drr_file(view_id, key)
            if drr_path is None:
                continue
            _, pixels = read_mha(str(drr_path))
            vn.drr_pixels[key] = pixels
            node = slicer.util.addVolumeFromArray(pixels[None].astype(np.float32), ijk_to_camworld,
                                                  f"DRR {view_id} {key}")
            node.SetAndObserveTransformNodeID(vn.carm.GetID())
            node.GetDisplayNode().AutoWindowLevelOn()
            vn.drrs[key] = self._own(node, sub)

        vn.composite = self._own(slicer.util.addVolumeFromArray(
            self._composite(vn)[None], ijk_to_camworld, f"X-ray {view_id}", "vtkMRMLVectorVolumeNode"), sub)
        vn.composite.SetAndObserveTransformNodeID(vn.carm.GetID())

        corners = cam.detector_point_camworld(cam.detector_corners_px())
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(*corners[0])      # outer corner of pixel (0, 0)
        plane.SetPoint1(*corners[1])      # along +u: texture s
        plane.SetPoint2(*corners[3])      # along +v: texture t (image rows, first stored row at t = 0)
        plane.Update()
        detector = slicer.modules.models.logic().AddModel(plane.GetOutput())
        detector.SetName(f"detector {view_id}")
        d = detector.GetDisplayNode()
        d.SetTextureImageDataConnection(vn.composite.GetImageDataConnection())
        d.SetColor(1.0, 1.0, 1.0)
        d.SetScalarVisibility(False)
        d.SetBackfaceCulling(False)
        d.SetAmbient(1.0)
        d.SetDiffuse(0.0)
        d.SetSpecular(0.0)
        d.SetInterpolateTexture(True)
        d.SetVisibility2D(False)
        detector.SetAndObserveTransformNodeID(vn.carm.GetID())
        vn.detector = self._own(detector, sub)

        pts_2d = run.points_2d(view_id)
        if pts_2d:
            names = list(pts_2d)
            vn.landmarks_2d = self._own(_markups(
                f"2D landmarks {view_id}", cam.detector_point_camworld(np.array([pts_2d[n] for n in names])),
                [_short(n) for n in names], names, LANDMARK_2D_COLOR, 6.0, vn.carm), sub)
        self.views[view_id] = vn

    def _composite(self, vn):
        overlays = []
        if self.drr_overlay and self.scene_state is not None and vn.view_id in self.scene_state.final_drr_views:
            for i, key in enumerate(self.run.target_keys):
                if key in vn.drr_pixels:
                    overlays.append((vn.drr_pixels[key], DRR_COLORS[i % len(DRR_COLORS)]))
        return composite_rgb(vn.xray_pixels, overlays, self.drr_opacity)

    # ------------------------------------------------------------------ state
    def stages_for(self, view_id, target_key):
        return self.run.pose_stages(view_id, target_key)

    def set_view(self, view_id):
        if view_id not in self.views:
            raise KeyError(f"no view {view_id!r}")
        defaults = default_stages(self.run, view_id)
        present = {k: self.run.pose_stages(view_id, k) for k in self.run.target_keys}
        self.stages = {k: (self.stages.get(k) if self.stages.get(k) in present[k] else defaults[k])
                       for k in self.run.target_keys}
        self.view_id = view_id
        self.update()

    def set_stage(self, target_key, stage):
        if stage not in self.run.pose_stages(self.view_id, target_key):
            raise ValueError(f"view {self.view_id} target {target_key} has no {stage} pose")
        self.stages[target_key] = stage
        self.update()

    def set_camera_centred(self, on):
        self.camera_centred = bool(on)
        self.update()

    def set_stages_all_views(self, on):
        self.stages_all_views = bool(on)
        self.update()

    def set_show_all_views(self, on):
        self.show_all_views = bool(on)
        self.update()

    def set_drr_overlay(self, on=None, opacity=None):
        if on is not None:
            self.drr_overlay = bool(on)
        if opacity is not None:
            self.drr_opacity = float(opacity)
        for vn in self.views.values():
            slicer.util.updateVolumeFromArray(vn.composite, self._composite(vn)[None])

    def set_models_visible(self, on):
        self.models_visible = bool(on)
        self.update()

    def set_landmarks_visible(self, on):
        """Shows or hides the 3D landmarks and the 2D landmarks on the detectors (their labels can cover the
        anatomy and the overlays)."""
        self.landmarks_visible = bool(on)
        self.update()

    def set_volume_rendering(self, on):
        if self.ct is None:
            return
        vr = slicer.modules.volumerendering.logic()
        display = vr.GetFirstVolumeRenderingDisplayNode(self.ct)
        if display is None:
            if not on:
                return
            display = vr.CreateDefaultVolumeRenderingNodes(self.ct)
            display.GetVolumePropertyNode().Copy(vr.GetPresetByName("CT-Bone"))
            if self.reused_ct is not None:
                self.nodes.append(display)  # the reused CT keeps no display node of this run
            self.nodes.append(display.GetVolumePropertyNode())
            if display.GetROINode() is not None:
                self.nodes.append(display.GetROINode())
        display.SetVisibility(bool(on))

    def update(self):
        """Applies the scene matrices of the current view, stages and frame to every node."""
        scene = scene_matrices(self.run, self.view_id, self.stages, self.camera_centred, self.stages_all_views)
        for key, node in self.anatomy.items():
            _set_matrix(node, scene.anatomy[key])
        for key, stage in scene.anatomy_stage.items():
            # A target without a pose in the selected view has no placement: hide it.
            if key in self.models:
                self.models[key].GetDisplayNode().SetVisibility(self.models_visible and stage is not None)
            if key in self.landmarks_3d:
                self.landmarks_3d[key].GetDisplayNode().SetVisibility(self.landmarks_visible and stage is not None)
        for view_id, vn in self.views.items():
            placed = view_id in scene.carm
            if placed:
                _set_matrix(vn.carm, scene.carm[view_id])
            selected = view_id == self.view_id
            visible = placed and (selected or self.show_all_views)
            vn.frustum.GetDisplayNode().SetColor(*(FRUSTUM_SELECTED_COLOR if selected else FRUSTUM_COLOR))
            for node in (vn.frustum, vn.detector):
                node.GetDisplayNode().SetVisibility(visible)
            if vn.landmarks_2d is not None:
                vn.landmarks_2d.GetDisplayNode().SetVisibility(visible and self.landmarks_visible)
        frame_changed = self.scene_state is not None and self.scene_state.frame != scene.frame
        before = self.scene_state.final_drr_views if self.scene_state is not None else set()
        self.scene_state = scene
        if before != scene.final_drr_views:
            for vn in self.views.values():
                slicer.util.updateVolumeFromArray(vn.composite, self._composite(vn)[None])
        self.show_detector_in_slice_view()
        if frame_changed:
            # everything moved to another world frame: re-centre the CT slice views and the 3D view
            lm = slicer.app.layoutManager()
            for name in (lm.sliceViewNames() if lm is not None else []):
                if name != "Red":
                    lm.sliceWidget(name).sliceLogic().FitSliceToAll()
            self.overview()

    # ------------------------------------------------------------------ views
    def pixel_to_world(self, view_id):
        """4x4 map from pixel index (u, v, 0) of a view to the world, read back from Slicer."""
        vn = self.views[view_id]
        world_from_parent = slicer.util.arrayFromTransformMatrix(vn.carm, toWorld=True)
        m = vtk.vtkMatrix4x4()
        vn.xray.GetIJKToRASMatrix(m)
        return world_from_parent @ slicer.util.arrayFromVTKMatrix(m)

    def show_detector_in_slice_view(self, view_id=None, slice_name="Red"):
        """Red slice view shows the view's X-ray + DRR composite as stored: u to the right, v down."""
        lm = slicer.app.layoutManager()
        widget = lm.sliceWidget(slice_name) if lm is not None else None
        if widget is None:
            return
        view_id = view_id or self.view_id
        vn = self.views[view_id]
        cam = self.run.camera(view_id)
        p = self.pixel_to_world(view_id)
        x = p[:3, 0] / np.linalg.norm(p[:3, 0])
        y = -p[:3, 1] / np.linalg.norm(p[:3, 1])
        slice_to_ras = np.eye(4)
        slice_to_ras[:3, 0], slice_to_ras[:3, 1], slice_to_ras[:3, 2] = x, y, np.cross(x, y)
        slice_to_ras[:3, 3] = apply(p, [(cam.cols - 1) / 2.0, (cam.rows - 1) / 2.0, 0.0])
        slice_node = widget.mrmlSliceNode()
        slice_node.GetSliceToRAS().DeepCopy(slicer.util.vtkMatrixFromArray(slice_to_ras))
        slice_node.UpdateMatrices()
        composite = widget.sliceLogic().GetSliceCompositeNode()
        composite.SetBackgroundVolumeID(vn.composite.GetID())
        composite.SetForegroundVolumeID(None)
        composite.SetLabelVolumeID(None)
        width = cam.cols * cam.col_spacing_mm
        height = cam.rows * cam.row_spacing_mm
        dims = slice_node.GetDimensions()
        aspect = dims[1] / float(dims[0]) if dims[0] > 0 else 1.0
        if height / width > aspect:
            fov_x, fov_y = height / aspect, height
        else:
            fov_x, fov_y = width, width * aspect
        slice_node.SetFieldOfView(fov_x * 1.02, fov_y * 1.02, slice_node.GetFieldOfView()[2])
        slice_node.SetXYZOrigin(0, 0, 0)

    def look_from_source(self, view_id=None, view_index=0):
        """Puts the 3D camera at the X-ray source looking at the detector centre (image rows down)."""
        view_id = view_id or self.view_id
        lm = slicer.app.layoutManager()
        if lm is None or lm.threeDWidget(view_index) is None:
            return
        cam = self.run.camera(view_id)
        vn = self.views[view_id]
        world_from_camworld = slicer.util.arrayFromTransformMatrix(vn.carm, toWorld=True)
        world_from_cam = world_from_camworld @ cam.camworld_from_cam
        source = apply(world_from_cam, [0.0, 0.0, 0.0])
        centre = apply(world_from_camworld, cam.detector_point_camworld(
            [(cam.cols - 1) / 2.0, (cam.rows - 1) / 2.0]))
        up = -world_from_cam[:3, 1]
        half_height = cam.rows * cam.row_spacing_mm / 2.0
        view = lm.threeDWidget(view_index).threeDView()
        camera_node = slicer.modules.cameras.logic().GetViewActiveCameraNode(view.mrmlViewNode())
        camera_node.SetPosition(*source)
        camera_node.SetFocalPoint(*centre)
        camera_node.SetViewUp(*up)
        camera_node.SetViewAngle(2.0 * np.degrees(np.arctan(1.05 * half_height / cam.source_to_detector_mm)))
        camera_node.ResetClippingRange()

    def overview(self, view_index=0, view_angle_deg=30.0):
        """3D camera beside the selected view's beam (from its +column side, raised toward image-up),
        framing every visible frustum and the anatomy."""
        lm = slicer.app.layoutManager()
        if lm is None or lm.threeDWidget(view_index) is None:
            return
        points = []
        for vn in self.views.values():
            if not vn.frustum.GetDisplayNode().GetVisibility():
                continue
            world = slicer.util.arrayFromTransformMatrix(vn.carm, toWorld=True)
            poly = vn.frustum.GetPolyData()
            points.extend(apply(world, poly.GetPoint(i)) for i in range(poly.GetNumberOfPoints()))
        for model in self.models.values():
            if model.GetDisplayNode().GetVisibility():
                bounds = [0.0] * 6
                model.GetRASBounds(bounds)
                points.extend([[bounds[0], bounds[2], bounds[4]], [bounds[1], bounds[3], bounds[5]]])
        if not points:
            return
        points = np.array(points)
        centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
        radius = float(np.max(np.linalg.norm(points - centre, axis=1)))
        cam = self.run.camera(self.view_id)
        world_from_cam = slicer.util.arrayFromTransformMatrix(
            self.views[self.view_id].carm, toWorld=True) @ cam.camworld_from_cam
        side, up = world_from_cam[:3, 0], -world_from_cam[:3, 1]
        direction = side + 0.6 * up
        direction /= np.linalg.norm(direction)
        view_up = up - np.dot(up, direction) * direction
        distance = radius / np.sin(np.radians(view_angle_deg / 2.0))
        view = lm.threeDWidget(view_index).threeDView()
        camera_node = slicer.modules.cameras.logic().GetViewActiveCameraNode(view.mrmlViewNode())
        camera_node.SetFocalPoint(*centre)
        camera_node.SetPosition(*(centre + distance * direction))
        camera_node.SetViewUp(*(view_up / np.linalg.norm(view_up)))
        camera_node.SetViewAngle(view_angle_deg)
        camera_node.ResetClippingRange()


class RunViewer:
    """Loads runs; the current RunScene is `self.scene` (one run at a time)."""

    def __init__(self):
        self.scene = None

    def load_run(self, path, load_ct=True, build_models=True, volume_rendering=False):
        run = XregRun.load(path)
        self.clear()
        scene = RunScene(run)
        try:
            scene.build(load_ct=load_ct, build_models=build_models, volume_rendering=volume_rendering)
        except Exception:
            scene.remove()
            raise
        self.scene = scene
        return scene

    def clear(self):
        if self.scene is not None:
            self.scene.remove()
            self.scene = None

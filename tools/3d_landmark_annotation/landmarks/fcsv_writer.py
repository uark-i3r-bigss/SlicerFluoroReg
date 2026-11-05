import os
import json
from pathlib import Path

import numpy as np

from . import status
from .model_lookup import find_model_by_name


def read_fcsv(path):
    """Read an FCSV file and return {label: np.array([x, y, z])}."""
    points = {}
    if not os.path.isfile(path):
        return points
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) < 12:
                continue
            try:
                coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
            points[parts[11]] = coords
    return points


def load_landmark_config():
    """Canonical landmark keys, output labels and visibility."""
    config_path = Path(__file__).parent.parent / "configs" / "landmark_config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as handle:
        return json.load(handle).get("landmarks", {})


def write_fcsv(landmarks, output_path, *, include_hidden=False, coordinate_system="LPS"):
    """Write landmarks to one FCSV.

    One writer, one rule: ``include_hidden`` decides whether landmarks marked
    ``"visible": false`` in configs/landmark_config.json are written.  Crop exports pass
    ``include_hidden=True`` because they are already filtered by crop membership; the
    canonical case FCSV does not.  ``DBG_*`` keys are never written.
    """
    landmark_config = load_landmark_config()
    header = [
        "# Markups fiducial file version = 5.8",
        f"# CoordinateSystem = {coordinate_system}",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID",
    ]
    rows = []
    for key, coords in landmarks.items():
        if key.startswith("DBG_"):
            continue
        entry = landmark_config.get(key, {})
        if not include_hidden and not entry.get("visible", True):
            continue
        if coords is None:
            print(f"  [WARN] Landmark {key} has no coordinate — skipping from FCSV")
            continue
        label = entry.get("label", key)
        x, y, z = (float(coords[0]), float(coords[1]), float(coords[2]))
        rows.append(
            f"vtkMRMLMarkupsFiducialNode_{len(rows)},{x:.6f},{y:.6f},{z:.6f},"
            f"0,0,0,1,1,1,0,{label},,vtkMRMLScalarVolumeNode1"
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(header + rows) + "\n")
    return output_path


def check_pelvis_models(totalseg_models):
    patterns = {
        "hip_left": "*hip_left.stl",
        "hip_right": "*hip_right.stl",
        "sacrum": "*sacrum.stl"
    }
    found = {}
    for key, pattern in patterns.items():
        path = find_model_by_name(totalseg_models, pattern)
        if path:
            found[key] = path
    
    # Need at least hips for APP
    return found if "hip_left" in found and "hip_right" in found else None


def check_femur_models(totalseg_models):
    patterns = {
        "femur_left": "*femur_left.stl",
        "femur_right": "*femur_right.stl"
    }
    found = {}
    for key, pattern in patterns.items():
        path = find_model_by_name(totalseg_models, pattern)
        if path:
            found[key] = path
    return found if len(found) > 0 else None


def check_lumbar_models(totalseg_models):
    patterns = {
        "L5": "*vertebrae_L5.stl",
        "L4": "*vertebrae_L4.stl",
        "L3": "*vertebrae_L3.stl",
        "L2": "*vertebrae_L2.stl",
        "L1": "*vertebrae_L1.stl",
        "sacrum": "*sacrum.stl"
    }
    found = {}
    for key, pattern in patterns.items():
        path = find_model_by_name(totalseg_models, pattern)
        if path:
            found[key] = path
    return found if len(found) > 0 else None


def detect_pelvis_landmarks_from_totalseg(totalseg_models):
    from .pelvis_landmarks import detect_pelvis_landmarks
    
    # Combined model is usually in the sibling 'models' directory
    combined_pelvis_path = find_model_by_name(
        os.path.join(Path(totalseg_models).parent.parent, "models"), 
        "*hip_left_hip_right.stl"
    )
    
    left_hip = find_model_by_name(totalseg_models, "*hip_left.stl")
    right_hip = find_model_by_name(totalseg_models, "*hip_right.stl")
    
    if combined_pelvis_path:
        return detect_pelvis_landmarks(combined_pelvis_path, left_hip, right_hip)
    elif left_hip and right_hip:
        print("  [INFO] No combined pelvis STL found — merging individual hip meshes for pelvis frame")
        return detect_pelvis_landmarks(None, left_hip, right_hip)
    else:
        return {}, {}


def detect_femur_landmarks_from_totalseg(totalseg_models, case_id=""):
    from .femur_landmarks import detect_femur_landmarks

    left_femur = find_model_by_name(totalseg_models, "*femur_left.stl")
    right_femur = find_model_by_name(totalseg_models, "*femur_right.stl")

    return detect_femur_landmarks(left_femur, right_femur, case_id=case_id)



# The acetabular socket center and the femoral head center are the same physical
# point, so every fitted femoral head center is also published in the pelvis
# object namespace.  Pelvis-only PnP then has well-separated geometry instead of
# the nearly coincident pubic/iliac points.  The pelvis entry is a verbatim copy
# of the femur entry and must never hold a different value.
PELVIS_FEMORAL_HEAD_SOURCES = {"PEL_FH_L": "FEM_L_FH", "PEL_FH_R": "FEM_R_FH"}


def add_pelvis_femoral_head_landmarks(landmarks):
    """Copy each available femoral head center into the pelvis namespace in place.

    A side without a femoral head center gets no pelvis point; nothing is
    recomputed, offset or substituted, so PEL_FH_L/PEL_FH_R always equal
    FEM_L_FH/FEM_R_FH exactly and share their LPS convention.
    """
    for pelvis_key, femur_key in PELVIS_FEMORAL_HEAD_SOURCES.items():
        coords = landmarks.get(femur_key)
        if coords is None:
            continue
        landmarks[pelvis_key] = coords
    return landmarks


def _load_mesh_report(models_dir):
    """Return the surface-generation record for this case, if the mesher wrote one.

    A landmark coordinate is uninterpretable without the surface that produced it, so the
    resolved MeshConfig and the per-mesh outcome travel with the landmarks.
    """
    case_root = Path(models_dir).parent
    for candidate in (case_root / "meta_data" / "mesh_report.json", case_root / "mesh_report.json"):
        if candidate.is_file():
            try:
                with open(candidate) as handle:
                    return json.load(handle)
            except (OSError, ValueError):
                return {"status": "unreadable", "path": str(candidate)}
    return {}




def _is_emittable(landmark_id, records):
    """A landmark reaches the FCSV only if its record carries a coordinate."""
    record = records.get(landmark_id)
    return record is None or record.emitted


def _pelvis_femoral_head_records(all_landmarks, records):
    """Records for the pelvis-namespace copies of the femoral head centres.

    The copy inherits the source's status exactly: if the femur fit abstained there is no
    pelvis point either, and the two namespaces can never disagree.
    """
    copies = {}
    for pelvis_key, femur_key in PELVIS_FEMORAL_HEAD_SOURCES.items():
        source = records.get(femur_key)
        if pelvis_key in all_landmarks and (source is None or source.emitted):
            copies[pelvis_key] = status.LandmarkRecord(
                landmark_id=pelvis_key,
                status=source.status if source else status.Status.OK,
                position=all_landmarks[pelvis_key],
                confidence=source.confidence if source else 1.0,
                reason=source.reason if source else None,
                residual_mm=source.residual_mm if source else None,
                support=source.support if source else 0,
                provenance=f"copy_of:{femur_key}",
                metrics={"copy_of": femur_key},
            )
        else:
            copies[pelvis_key] = status.abstained(
                pelvis_key,
                source.reason if source else status.Reason.MISSING_INPUT,
                provenance=f"copy_of:{femur_key}",
                metrics={"copy_of": femur_key},
            )
    return copies


def _complete_records(records):
    """Add a record for every configured landmark the detectors never reported on."""
    complete = dict(records)
    for landmark_id in load_landmark_config():
        complete.setdefault(
            landmark_id,
            status.not_attempted(landmark_id, status.Reason.NOT_CONFIGURED,
                                 provenance="landmark_config"),
        )
    return complete


def create_comprehensive_landmark_file(models_dir, output_dir, case_id):
    
    os.makedirs(output_dir, exist_ok=True)
    
    totalseg_models = Path(models_dir).parent / "TotalSegmentator_outputs" / "models"
    if not totalseg_models.exists():
        totalseg_models = Path(models_dir)
    
    all_landmarks = {}
    records = {}
    debug_data = {
        'spheres': {},
        'planes': {},
        'models_used': {},
        'mesh': _load_mesh_report(models_dir),
    }

    pelvis_detected_count = 0
    pelvis_available = check_pelvis_models(str(totalseg_models))
    if pelvis_available:
        print("  Pelvis models found - detecting pelvis landmarks...")
        pelvis_landmarks, pelvis_debug = detect_pelvis_landmarks_from_totalseg(str(totalseg_models))
        all_landmarks.update(pelvis_landmarks)
        records.update(pelvis_debug.get('records', {}))
        pelvis_detected_count = len([k for k in pelvis_landmarks if k.startswith("PEL_")])
        debug_data['models_used']['pelvis'] = pelvis_available
        if 'APP_plane' in pelvis_debug:
            debug_data['planes']['APP'] = pelvis_debug['APP_plane']

    femur_available = check_femur_models(str(totalseg_models))
    if femur_available:
        print(f"  Femur models found ({', '.join(femur_available.keys())}) - detecting femur landmarks...")
        femur_landmarks, femur_debug_info = detect_femur_landmarks_from_totalseg(
            str(totalseg_models), case_id=case_id
        )
        all_landmarks.update(femur_landmarks)
        records.update(femur_debug_info.get('records', {}))
        debug_data['models_used']['femur'] = femur_available
        if 'FEM_L_FH' in femur_debug_info:
            debug_data['spheres']['FEM_L_FH'] = femur_debug_info['FEM_L_FH']
        if 'FEM_R_FH' in femur_debug_info:
            debug_data['spheres']['FEM_R_FH'] = femur_debug_info['FEM_R_FH']
        if 'FP_L' in femur_debug_info:
            debug_data['planes']['FP_L'] = femur_debug_info['FP_L']
        if 'FP_R' in femur_debug_info:
            debug_data['planes']['FP_R'] = femur_debug_info['FP_R']

    add_pelvis_femoral_head_landmarks(all_landmarks)
    records.update(_pelvis_femoral_head_records(all_landmarks, records))
    mirrored = [k for k in PELVIS_FEMORAL_HEAD_SOURCES if k in all_landmarks]
    if mirrored:
        print(f"  Pelvis femoral head centers copied from the femur landmarks: {', '.join(sorted(mirrored))}")

    lumbar_available = check_lumbar_models(str(totalseg_models))
    if lumbar_available:
        print("  Lumbar models found - detecting lumbar landmarks...")
        from .lumbar_landmarks import generate_lumbar_landmarks
        lumbar_landmarks, lumbar_records = generate_lumbar_landmarks(
            str(totalseg_models), with_records=True
        )
        all_landmarks.update(lumbar_landmarks)
        records.update(lumbar_records)
        debug_data['models_used']['lumbar'] = lumbar_available
    
    pelvis_count = len([k for k in all_landmarks if k.startswith("PEL_") and not k.startswith("DBG_")])
    femur_count = len([k for k in all_landmarks if k.startswith("FEM_")])
    lumbar_count = len([k for k in all_landmarks if k.startswith("LUM_")])
    landmark_counts = {"pelvis": pelvis_count, "femur": femur_count, "lumbar": lumbar_count}

    if pelvis_available and pelvis_detected_count == 0:
        print("  [WARN] Pelvis models were found but no pelvis landmarks were generated")
    if femur_available and femur_count == 0:
        print("  [WARN] Femur models were found but no femur landmarks were generated")
    if lumbar_available and lumbar_count == 0:
        print("  [WARN] Lumbar models were found but no lumbar landmarks were generated")

    print(f"  Landmark counts: pelvis={pelvis_count}, femur={femur_count}, lumbar={lumbar_count}")

    # Only records whose status carries a coordinate reach the FCSV; every other landmark the
    # registry knows about is still reported in the debug JSON with its status and reason.
    emitted = {key: value for key, value in all_landmarks.items()
               if key.startswith("DBG_") or _is_emittable(key, records)}
    abstained = sorted(key for key, record in records.items()
                       if record.status not in (status.Status.OK, status.Status.LOW_CONFIDENCE))
    if abstained:
        print(f"  Abstained or unavailable ({len(abstained)}): {', '.join(abstained)}")

    debug_data['landmarks'] = status.records_to_json(_complete_records(records))

    output_filename = f"{case_id}_landmarks.fcsv"
    output_path = os.path.join(output_dir, output_filename)
    fcsv_path = write_fcsv(emitted, output_path)

    debug_filename = f"{case_id}_debug.json"
    debug_path = os.path.join(output_dir, debug_filename)

    def json_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(debug_path, 'w') as f:
        json.dump(debug_data, f, indent=2, default=json_serializer)

    print(f"  Saved landmarks: {output_path}")
    print(f"  Saved debug data: {debug_path}")

    return fcsv_path, landmark_counts

"""X-ray steps: ``import-xrays`` (DICOM copy or image wrapping with ids, then conversion) and ``convert-xrays``."""

import os
import re

from .project import copy_file, read_json, rel, write_json
from .runner import new_log_path, run_tool
from .status import InputError, ToolError, emit

DICOM_DIR = "meta_data/XrayDICOM"
PROJ_DIR = "meta_data/XrayProjData"
INDEX_NAME = "xray_index.json"
CONVERTER_REL = "apps/image_io/convert_radiograph_dicom_to_proj_data/xreg-convert-dicom-radiograph"
# Native grid: keep the stored pixels and the metadata spacing, no Bayview extrinsic special case.
CONVERTER_FLAGS = ("--no-proc", "--no-bayview-check", "--no-guess-spacing", "--pixel-type", "float")
# The converter picks the camera frame from Modality (XA/RF: det-neg-z, the C-arm frame every later step expects;
# CR/DX: det-pos-z) and treats a missing Modality like DX. C-arm exports without the tag get det-neg-z.
MISSING_MODALITY_FLAGS = ("--proj-frame", "det-neg-z")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def natural_key(name):
    """Sort key ordering embedded numbers numerically: ``2_1`` before ``10_1``."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def is_dicom(path):
    try:
        with open(path, "rb") as handle:
            head = handle.read(132)
    except OSError:
        return False
    return len(head) == 132 and head[128:132] == b"DICM"


def is_image(path):
    return os.path.splitext(path)[1].lower() in IMAGE_SUFFIXES


def dicom_modality(path):
    """Value of (0008,0060) Modality, '' when absent. Reads only the file's start (group 0008 comes first)."""
    with open(path, "rb") as handle:
        data = handle.read(1 << 16)
    at = data.find(b"\x08\x00\x60\x00", 132)
    if at < 0:
        return ""
    if data[at + 4:at + 6] == b"CS":  # explicit VR little endian: 2-byte length
        size, start = int.from_bytes(data[at + 6:at + 8], "little"), at + 8
    else:  # implicit VR little endian: 4-byte length
        size, start = int.from_bytes(data[at + 4:at + 8], "little"), at + 8
    if size > 64:
        return ""
    return data[start:start + size].decode("ascii", "replace").strip("\x00 ").upper()


def format_id(number):
    return f"{number:03d}"


def index_path(project):
    return os.path.join(project, *DICOM_DIR.split("/"), INDEX_NAME)


def load_index(project):
    path = index_path(project)
    if not os.path.isfile(path):
        return {}
    data = read_json(path, "X-ray index")
    if not isinstance(data, dict):
        raise InputError(f"{path}: expected an object of ids")
    return data


def _input_files(inputs):
    """Every input file in order: a folder contributes its files in natural name order."""
    files = []
    for raw in inputs:
        path = os.path.abspath(os.path.expanduser(raw))
        if os.path.isdir(path):
            names = sorted((n for n in os.listdir(path) if os.path.isfile(os.path.join(path, n))), key=natural_key)
            files += [os.path.join(path, n) for n in names]
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise InputError(f"--input {path} does not exist")
    return files


def _image_libraries():
    try:
        import numpy
        import pydicom
        from PIL import Image
    except ImportError as exc:
        raise InputError(f"image files need pydicom, numpy and Pillow ({exc}); run the command with Slicer's "
                         "PythonSlicer or install them")
    return numpy, pydicom, Image


def wrap_image(image_path, template_path, out_path):
    """Write ``image_path`` as a DICOM: a copy of the template (geometry and all other attributes) with the
    image as grayscale uint16 pixels resized (bilinear) to the template's Rows x Columns and new UIDs."""
    np, pydicom, Image = _image_libraries()
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = pydicom.dcmread(template_path)
    rows, cols = int(ds.Rows), int(ds.Columns)
    with Image.open(image_path) as img:
        if img.mode.startswith("I;16"):
            img = img.convert("I")
        elif img.mode not in ("I", "F", "L"):
            img = img.convert("L")
        img = img.convert("F")
        if img.size != (cols, rows):
            img = img.resize((cols, rows), Image.BILINEAR)
        pixels = np.clip(np.rint(np.asarray(img, dtype=np.float64)), 0, 65535).astype("<u2")
    ds.SamplesPerPixel = 1
    if ds.get("PhotometricInterpretation") not in ("MONOCHROME1", "MONOCHROME2"):
        ds.PhotometricInterpretation = "MONOCHROME2"
    if "PlanarConfiguration" in ds:
        del ds.PlanarConfiguration
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    ds["PixelData"].VR = "OW"
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    tmp = out_path + ".tmp"
    ds.save_as(tmp, write_like_original=False)
    os.replace(tmp, out_path)


def import_xrays(project, inputs, settings, convert=True):
    """Append the input X-rays as the next ids (input order), then convert them unless ``convert`` is False."""
    dicom_dir = os.path.join(project, *DICOM_DIR.split("/"))
    if not os.path.isdir(dicom_dir):
        raise InputError(f"{project}: missing {DICOM_DIR}; run init first")
    files = _input_files(inputs)
    kinds = [(f, "dicom" if is_dicom(f) else "image" if is_image(f) else None) for f in files]
    skipped = [os.path.basename(f) for f, kind in kinds if kind is None]
    kinds = [(f, kind) for f, kind in kinds if kind]
    if not kinds:
        raise InputError("no DICOM (DICM magic at byte 128) or image files ("
                         + " ".join(IMAGE_SUFFIXES) + ") in --input")
    if skipped:
        emit("warning", "import-xrays", message=f"{len(skipped)} other file(s) ignored: {', '.join(skipped)}")
    template = None
    if any(kind == "image" for _, kind in kinds):
        if not settings.get("xray_template"):
            raise InputError("image files need a template DICOM from the same C-arm: --xray-template FILE "
                             "(or 'xray_template' in the settings file)")
        template = settings.require("xray_template")
        if not is_dicom(template):
            raise InputError(f"--xray-template {template} is not a DICOM file")
        _image_libraries()
    if convert:
        converter(settings)

    index = load_index(project)
    present = [n for n in os.listdir(dicom_dir) if n.endswith(".dcm")]
    numbers = [int(i) for i in index if i.isdigit()]
    numbers += [int(n[:-4]) for n in present if n[:-4].isdigit()]
    next_number = max(numbers, default=0) + 1
    imported = []
    try:
        for k, (path, kind) in enumerate(kinds):
            xid = format_id(next_number + k)
            out = os.path.join(dicom_dir, xid + ".dcm")
            if kind == "dicom":
                copy_file(path, out)
            else:
                wrap_image(path, template, out)
            index[xid] = {"source_name": os.path.basename(path)}
            imported.append(xid)
            emit("progress", "import-xrays", phase="import", id=xid, done=k + 1, total=len(kinds),
                 message=os.path.basename(path))
    finally:
        write_json(index_path(project), dict(sorted(index.items(), key=lambda kv: natural_key(kv[0]))))
    result = {"ids": imported, "index": f"{DICOM_DIR}/{INDEX_NAME}",
              "sources": {xid: index[xid]["source_name"] for xid in imported}, "converted": [], "failed": []}
    if convert:
        converted = _convert_ids(project, converter(settings), imported, overwrite=True, step="import-xrays")
        result.update({"converted": converted["converted"], "failed": converted["failed"],
                       "logs": converted["logs"]})
    return result


def dicom_ids(project):
    dicom_dir = os.path.join(project, *DICOM_DIR.split("/"))
    if not os.path.isdir(dicom_dir):
        return []
    return sorted((n[:-4] for n in os.listdir(dicom_dir) if n.endswith(".dcm")), key=natural_key)


def converted_ids(project):
    proj_dir = os.path.join(project, *PROJ_DIR.split("/"))
    if not os.path.isdir(proj_dir):
        return []
    return sorted((n[:-3] for n in os.listdir(proj_dir) if n.endswith(".h5")), key=natural_key)


def converter(settings):
    exe = os.path.join(settings.require("xreg_build_dir"), *CONVERTER_REL.split("/"))
    if not os.path.isfile(exe) or not os.access(exe, os.X_OK):
        raise InputError(f"converter not found: {exe} (check --xreg-build-dir / --xreg-dir)")
    return exe


def _convert_ids(project, exe, ids, overwrite, step):
    proj_dir = os.path.join(project, *PROJ_DIR.split("/"))
    os.makedirs(proj_dir, exist_ok=True)
    converted, skipped, failed, logs = [], [], [], {}
    for k, xid in enumerate(ids):
        out = os.path.join(proj_dir, xid + ".h5")
        if os.path.exists(out) and not overwrite:
            skipped.append(xid)
            emit("progress", step, phase="convert", id=xid, done=k + 1, total=len(ids), message="exists, skipped")
            continue
        # Relative to the experiment (the converter runs there): the projection file records this path and xreg
        # resolves it against the run's working directory, the experiment, so a moved experiment stays complete.
        dcm = f"{DICOM_DIR}/{xid}.dcm"
        log_path = new_log_path(project, "project", "convert-xrays", suffix=xid)
        logs[xid] = rel(project, log_path)
        if os.path.exists(out):
            os.remove(out)
        frame = MISSING_MODALITY_FLAGS if not dicom_modality(os.path.join(project, *dcm.split("/"))) else ()
        rc = run_tool([exe] + list(CONVERTER_FLAGS) + list(frame) + [dcm, out, "-v"], cwd=project,
                      log_path=log_path)
        if rc != 0 or not os.path.isfile(out):
            failed.append(xid)
            emit("error", step, id=xid, message=f"conversion failed (status {rc}); see {logs[xid]}")
        else:
            converted.append(xid)
            emit("progress", step, phase="convert", id=xid, done=k + 1, total=len(ids), message="converted")
    return {"converted": [f"{PROJ_DIR}/{i}.h5" for i in converted], "skipped": skipped, "failed": failed,
            "logs": logs}


def convert_xrays(project, settings, ids=None, overwrite=False):
    exe = converter(settings)
    available = dicom_ids(project)
    ids = list(ids) if ids else available
    missing = [i for i in ids if i not in available]
    if missing:
        raise InputError(f"no DICOM in {DICOM_DIR} for id(s) {', '.join(missing)}")
    if not ids:
        raise InputError(f"no X-rays in {DICOM_DIR}; run import-xrays first")
    result = _convert_ids(project, exe, ids, overwrite, step="convert-xrays")
    if result["failed"]:
        raise ToolError(f"{len(result['failed'])} conversion(s) failed: {', '.join(result['failed'])}", result)
    return result

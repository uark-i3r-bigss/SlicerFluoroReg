#!/usr/bin/env python3
"""Builds the Slicer extension package installed with Extensions Manager -> "Install from file...".

    python3 scripts/package_extension.py [--slicer-version X.Y] [--out DIR]

Writes <DIR>/<name>-linux-x86_64-Slicer<X.Y>-git<shortsha>[-dirty].tar.gz (default DIR: dist/) with the layout of
an official Linux scripted-extension package:

    <top>/lib/Slicer-X.Y/qt-scripted-modules/<module files>
    <top>/share/Slicer-X.Y/<name>.s4ext
    <top>/share/Slicer-X.Y/<name>/{LICENSE,THIRD_PARTY_NOTICES,README.md,docs/*,tools/*,RELEASE_MANIFEST.json}

<name> and the s4ext values come from the top-level CMakeLists.txt (project() and EXTENSION_*; scmurl is the
homepage, scmrevision the commit). The modules are the add_subdirectory(<module>) entries of the top-level
CMakeLists.txt; the files of each module are the SCRIPTS and RESOURCES listed in <module>/CMakeLists.txt. They
and every CMakeLists.txt must be tracked by git, every tracked file of a module directory must be listed, and no
two modules may install the same path. Required release documents must also be tracked. In addition to the
module payload, the package installs those documents, curated backend sources listed in
tools/VENDORED_COMPONENTS.json, and a generated source/payload manifest. Files are from the
working tree; "-dirty" marks uncommitted changes to tracked files.
X.Y is the Slicer version the package is for (Slicer loads lib/Slicer-<its X.Y>/qt-scripted-modules):
--slicer-version, else read from the Slicer installation (lib/Slicer-X.Y) of $SLICER, else of Slicer on PATH.
Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
_REQUIRED_KEYS = ("EXTENSION_HOMEPAGE", "EXTENSION_CATEGORY", "EXTENSION_CONTRIBUTORS", "EXTENSION_DESCRIPTION",
                  "EXTENSION_ICONURL", "EXTENSION_SCREENSHOTURLS", "EXTENSION_DEPENDS")
_DOCUMENT_FILES = ("LICENSE", "THIRD_PARTY_NOTICES", "README.md",
                   "docs/setup.md", "docs/pipeline.md", "docs/developer.md")
_VENDOR_MANIFEST = "tools/VENDORED_COMPONENTS.json"
_XREG_REVISION = "0b2e0dc9aa7598d49cb634eed931b769fc9918dd"
_FORBIDDEN_VENDOR_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "tests", "test",
                           "agents", ".agents", "node_modules", "weights", "data",
                           "datasets", "images"}
_FORBIDDEN_VENDOR_SUFFIXES = {".pyc", ".pyo", ".pth", ".pt", ".ckpt", ".nii", ".gz",
                              ".dcm", ".mha", ".mhd", ".nrrd", ".fcsv", ".stl", ".obj", ".ply",
                              ".vtk", ".vtp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif",
                              ".mp4", ".zip"}


class PackageError(RuntimeError):
    pass


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(^|\s)#.*$", "", line) for line in text.splitlines())


def extension_metadata(repo: Path) -> Dict[str, str]:
    """{"name": project name, "EXTENSION_*": value} of the top-level CMakeLists.txt."""
    text = (repo / "CMakeLists.txt").read_text(encoding="utf-8")
    project = re.search(r"^\s*project\(\s*(\w+)", text, re.MULTILINE)
    if project is None:
        raise PackageError(f"{repo / 'CMakeLists.txt'}: no project(<name>)")
    meta = {"name": project.group(1)}
    for key, value in re.findall(r'^\s*set\(\s*(EXTENSION_\w+)\s+"([^"]*)"\s*\)', text, re.MULTILINE):
        meta[key] = value
    missing = [k for k in _REQUIRED_KEYS if k not in meta]
    if missing:
        raise PackageError(f"{repo / 'CMakeLists.txt'}: missing set({', '.join(missing)} \"...\")")
    return meta


def module_names(repo: Path) -> List[str]:
    """Module directories added with add_subdirectory(<module>) in the top-level CMakeLists.txt, in order."""
    text = _strip_comments((repo / "CMakeLists.txt").read_text(encoding="utf-8"))
    names = re.findall(r"\badd_subdirectory\(\s*([^\s)]+)\s*\)", text)
    if not names:
        raise PackageError(f"{repo / 'CMakeLists.txt'}: no add_subdirectory(<module>)")
    return names


def module_files(repo: Path, module: str) -> List[str]:
    """Files listed in <module>/CMakeLists.txt (SCRIPTS and RESOURCES), relative to the module dir."""
    cmake = repo / module / "CMakeLists.txt"
    if not cmake.is_file():
        raise PackageError(f"{module}/CMakeLists.txt not found")
    text = _strip_comments(cmake.read_text(encoding="utf-8"))
    sets = {name: body.split() for name, body in re.findall(r"\bset\(\s*(\w+)\s+([^)]*)\)", text)}
    name = sets.get("MODULE_NAME", [None])[0]
    if name != module:
        raise PackageError(f"{module}/CMakeLists.txt: MODULE_NAME is {name!r}, expected {module!r}")
    files = [f.replace("${MODULE_NAME}", name)
             for f in sets.get("MODULE_PYTHON_SCRIPTS", []) + sets.get("MODULE_PYTHON_RESOURCES", [])]
    if f"{name}.py" not in files:
        raise PackageError(f"{module}/CMakeLists.txt does not list {name}.py")
    return files


def _git(repo: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        raise PackageError(f"git {' '.join(args)} failed in {repo}: {e.stderr.strip()}") from None
    return out.stdout


def packaged_files(repo: Path) -> List[Tuple[str, str]]:
    """(module, file relative to the module dir) of every module, after checking them and the CMakeLists.txt
    files against git's tracked files and checking that no two modules install the same path."""
    tracked = {p for p in _git(repo, "ls-files", "-z").split("\0") if p}
    files: List[Tuple[str, str]] = []
    owner: Dict[str, str] = {}
    for module in module_names(repo):
        listed = module_files(repo, module)
        required = [f"{module}/CMakeLists.txt"] + [f"{module}/{f}" for f in listed]
        untracked = [f for f in ["CMakeLists.txt"] + required if f not in tracked]
        if untracked:
            raise PackageError(f"not tracked by git (commit them first): {', '.join(untracked)}")
        unlisted = sorted(p for p in tracked - set(required) if p.startswith(f"{module}/"))
        if unlisted:
            raise PackageError(f"tracked but not listed in {module}/CMakeLists.txt: {', '.join(unlisted)}")
        for rel in listed:
            if rel in owner:
                raise PackageError(f"path collision: {rel} is installed by both {owner[rel]} and {module}")
            owner[rel] = module
            files.append((module, rel))
    return files


def vendor_files(repo: Path) -> Tuple[List[str], Dict[str, dict]]:
    """Validate the committed allowlist and return curated files relative to tools/."""
    tracked = {p for p in _git(repo, "ls-files", "-z").split("\0") if p}
    path = repo / _VENDOR_MANIFEST
    if _VENDOR_MANIFEST not in tracked or not path.is_file():
        raise PackageError(f"required vendor manifest missing or not tracked: {_VENDOR_MANIFEST}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise PackageError(f"invalid vendor manifest: {exc}") from None
    components = manifest.get("components", {})
    if manifest.get("schema_version") != 1 or set(components) != {"pose", "annotation"}:
        raise PackageError("vendor manifest requires schema_version 1 and pose/annotation components")
    files = ["VENDORED_COMPONENTS.json"]
    for name, component in sorted(components.items()):
        directory = component.get("directory", "")
        if not directory or Path(directory).is_absolute() or ".." in Path(directory).parts:
            raise PackageError(f"invalid vendor directory: {directory!r}")
        hashes = component.get("files", {})
        if not hashes or not component.get("source_revision") or not component.get("license"):
            raise PackageError(f"incomplete vendor component: {name}")
        for rel, digest in sorted(hashes.items()):
            relative = Path(directory) / rel
            if (Path(rel).is_absolute() or ".." in relative.parts
                    or any(part.lower() in _FORBIDDEN_VENDOR_PARTS for part in relative.parts)
                    or relative.suffix.lower() in _FORBIDDEN_VENDOR_SUFFIXES):
                raise PackageError(f"forbidden vendor payload: {relative}")
            source = repo / "tools" / relative
            if source.is_symlink() or not source.is_file():
                raise PackageError(f"vendor payload must be an ordinary file: {relative}")
            if source.relative_to(repo).as_posix() not in tracked:
                raise PackageError(f"vendor payload not tracked: {relative}")
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise PackageError(f"vendor payload hash mismatch: {relative}")
            files.append(relative.as_posix())
        snapshot = hashlib.sha256("".join(f"{p}\0{h}\n" for p, h in sorted(hashes.items())).encode()).hexdigest()
        if component.get("snapshot_sha256") != snapshot:
            raise PackageError(f"vendor snapshot hash mismatch: {name}")
    if len(files) != len(set(files)):
        raise PackageError("vendor components contain overlapping paths")
    extra = sorted(p for p in tracked if p.startswith("tools/") and p[6:] not in files)
    if extra:
        raise PackageError(f"tracked tools files missing from vendor allowlist: {', '.join(extra)}")
    return sorted(files), components


def component_inventory(components: Dict[str, dict]) -> Dict[str, dict]:
    inventory = {name: {"bundled": True, "kind": "source", **{
        key: value for key, value in component.items() if key != "files"}}
        for name, component in components.items()}
    inventory.update({
        "xreg": {"bundled": False, "kind": "manual", "access": "permission-required",
                 "availability": "maintainer-provided", "source_revision": _XREG_REVISION},
        "environments": {"bundled": False, "kind": "user-runtime"},
        "weights": {"bundled": False, "kind": "upstream-download"},
    })
    return inventory


def find_slicer() -> Path:
    """The Slicer executable: $SLICER, else Slicer on PATH (symbolic links resolved)."""
    exe = os.environ.get("SLICER") or shutil.which("Slicer")
    if not exe:
        raise PackageError("Slicer not found: set SLICER to the Slicer executable (<Slicer install dir>/Slicer) "
                           "or pass --slicer-version X.Y")
    path = Path(exe).resolve()
    if not (path.is_file() and os.access(str(path), os.X_OK)):
        raise PackageError(f"SLICER={exe} is not an executable file")
    return path


def slicer_version_of(slicer: Path) -> str:
    """X.Y of a Slicer installation, from its lib/Slicer-X.Y directory."""
    found = sorted(p.name[len("Slicer-"):] for p in (slicer.parent / "lib").glob("Slicer-*")
                   if p.is_dir() and re.fullmatch(r"Slicer-\d+\.\d+", p.name))
    if len(found) != 1:
        raise PackageError(f"expected one lib/Slicer-X.Y in {slicer.parent}, found {found or 'none'}")
    return found[0]


def s4ext_text(meta: Dict[str, str], revision: str) -> str:
    fields = [("scm", "git"), ("scmurl", meta["EXTENSION_HOMEPAGE"]), ("scmrevision", revision),
              ("depends", meta["EXTENSION_DEPENDS"]), ("build_subdirectory", "."),
              ("homepage", meta["EXTENSION_HOMEPAGE"]), ("contributors", meta["EXTENSION_CONTRIBUTORS"]),
              ("category", meta["EXTENSION_CATEGORY"]), ("iconurl", meta["EXTENSION_ICONURL"]),
              ("status", meta.get("EXTENSION_STATUS", "")), ("description", meta["EXTENSION_DESCRIPTION"]),
              ("screenshoturls", meta["EXTENSION_SCREENSHOTURLS"]), ("enabled", "1")]
    return "".join(f"{k} {v}".rstrip() + "\n" for k, v in fields)


def release_payload(repo: Path, slicer_version: str) -> Tuple[Dict[str, str], str, bool, int, list]:
    """Assemble one validated payload shared by the local and CMake/CPack package paths."""
    if not re.fullmatch(r"\d+\.\d+", slicer_version):
        raise PackageError(f"Slicer version {slicer_version!r} is not X.Y")
    meta = extension_metadata(repo)
    files = packaged_files(repo)
    vendors, components = vendor_files(repo)
    tracked = set(_git(repo, "ls-files", "-z").split("\0"))
    missing_docs = [p for p in _DOCUMENT_FILES if p not in tracked or not (repo / p).is_file()]
    if missing_docs:
        raise PackageError(f"required release documents missing or not tracked: {', '.join(missing_docs)}")
    revision = _git(repo, "rev-parse", "HEAD").strip()
    dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no").strip())
    mtime = int(_git(repo, "log", "-1", "--format=%ct").strip())
    module_dir = f"lib/Slicer-{slicer_version}/qt-scripted-modules"
    doc_dir = f"share/Slicer-{slicer_version}/{meta['name']}"
    sources = [(f"{module_dir}/{rel}", repo / module / rel) for module, rel in files]
    sources.extend((f"{doc_dir}/{rel}", repo / rel) for rel in _DOCUMENT_FILES)
    sources.extend((f"{doc_dir}/tools/{rel}", repo / "tools" / rel) for rel in vendors)
    payload = [(dest, source.read_bytes(), 0o755 if source.stat().st_mode & 0o111 else 0o644)
               for dest, source in sources]
    manifest = {
        "schema_version": 1,
        "extension": meta["name"],
        "release_version": meta.get("EXTENSION_VERSION", ""),
        "source_url": meta["EXTENSION_HOMEPAGE"],
        "source_revision": revision,
        "source_dirty": dirty,
        "platform": "linux-x86_64",
        "slicer_version": slicer_version,
        "modules": module_names(repo),
        "setup_document": f"{doc_dir}/docs/setup.md",
        "components": component_inventory(components),
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                  for name, data, mode in payload},
    }
    payload.append((f"{doc_dir}/RELEASE_MANIFEST.json",
                    (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"), 0o644))
    return meta, revision, dirty, mtime, payload


def cmake_support(repo: Path, slicer_version: str, output: Path) -> None:
    """Generate explicit vendor install rules and the matching CPack payload manifest."""
    meta, revision, dirty, mtime, payload = release_payload(repo, slicer_version)
    manifest_name, manifest_data, _ = payload[-1]
    manifest_path = output.with_name("RELEASE_MANIFEST.json")
    manifest_path.write_bytes(manifest_data)
    lines = []
    vendors, _ = vendor_files(repo)
    doc_dir = f"share/Slicer-{slicer_version}/{meta['name']}"
    for rel in vendors:
        source = repo / "tools" / rel
        destination = f"{doc_dir}/tools/{Path(rel).parent.as_posix()}"
        command = "PROGRAMS" if source.stat().st_mode & 0o111 else "FILES"
        lines.append(f'install({command} "{source.as_posix()}" DESTINATION "{destination}" COMPONENT Runtime)')
    lines.append(f'install(FILES "{manifest_path.as_posix()}" DESTINATION "{doc_dir}" COMPONENT Runtime)')
    dependencies = [repo / module / rel for module, rel in packaged_files(repo)]
    dependencies.extend(repo / rel for rel in _DOCUMENT_FILES)
    dependencies.extend(repo / "tools" / rel for rel in vendors)
    dependencies.append(repo / "scripts/package_extension.py")
    lines.append("set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS")
    lines.extend(f'  "{path.as_posix()}"' for path in dependencies)
    lines.append(")")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(repo: Path, slicer_version: str, out_dir: Path) -> Path:
    """Writes the package and returns its path."""
    meta, revision, dirty, mtime, payload = release_payload(repo, slicer_version)
    top = f"{meta['name']}-linux-x86_64-Slicer{slicer_version}-git{revision[:7]}{'-dirty' if dirty else ''}"

    s4ext_name = f"share/Slicer-{slicer_version}/{meta['name']}.s4ext"
    s4ext_data = s4ext_text(meta, revision).encode("utf-8")
    manifest_name, manifest_data, manifest_mode = payload.pop()
    manifest = json.loads(manifest_data)
    manifest["files"][s4ext_name] = {"sha256": hashlib.sha256(s4ext_data).hexdigest(), "size": len(s4ext_data)}
    payload.extend([(s4ext_name, s4ext_data, 0o644),
                    (manifest_name, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"), manifest_mode)])
    def add(tar: tarfile.TarFile, name: str, data: bytes, mode: int) -> None:
        info = tarfile.TarInfo(f"{top}/{name}")
        info.size, info.mtime, info.mode = len(data), mtime, mode
        tar.addfile(info, io.BytesIO(data))

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{top}.tar.gz"
    tmp = archive.with_name(archive.name + ".tmp")
    with tarfile.open(str(tmp), "w:gz", format=tarfile.GNU_FORMAT) as tar:
        for name, data, mode in payload:
            add(tar, name, data, mode)
    tmp.replace(archive)
    return archive


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slicer-version", metavar="X.Y",
                        help="Slicer version the package is for (default: that of $SLICER or Slicer on PATH)")
    parser.add_argument("--out", type=Path, default=REPO / "dist", help="output directory (default: dist/)")
    parser.add_argument("--cmake-support", type=Path, metavar="FILE",
                        help="generate CMake vendor install rules and CPack payload manifest")
    args = parser.parse_args(argv)
    try:
        version = args.slicer_version or slicer_version_of(find_slicer())
        if args.cmake_support:
            cmake_support(REPO, version, args.cmake_support)
            return 0
        archive = build_package(REPO, version, args.out)
    except PackageError as e:
        print(f"package_extension: {e}", file=sys.stderr)
        return 1
    print(archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())

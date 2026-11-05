"""Minimal MetaImage (.mha, data LOCAL) reader: header, dimensions and pixels, no ITK needed."""
from __future__ import annotations

import zlib
from typing import Dict, Tuple

import numpy as np

_ELEMENT_TYPES = {
    "MET_UCHAR": "u1", "MET_CHAR": "i1", "MET_USHORT": "u2", "MET_SHORT": "i2",
    "MET_UINT": "u4", "MET_INT": "i4", "MET_ULONG_LONG": "u8", "MET_LONG_LONG": "i8",
    "MET_FLOAT": "f4", "MET_DOUBLE": "f8",
}
_MAX_HEADER_BYTES = 1 << 16


class MhaError(ValueError):
    pass


def _parse_header(raw: bytes, path) -> Tuple[Dict[str, str], int]:
    header: Dict[str, str] = {}
    pos = 0
    while True:
        end = raw.find(b"\n", pos)
        if end < 0:
            raise MhaError(f"{path}: MetaImage header has no ElementDataFile line")
        line = raw[pos:end].decode("ascii", errors="replace").strip()
        pos = end + 1
        key, sep, value = line.partition("=")
        if not sep:
            continue
        header[key.strip()] = value.strip()
        if key.strip() == "ElementDataFile":
            return header, pos


def read_mha_header(path) -> Dict[str, str]:
    with open(path, "rb") as f:
        raw = f.read(_MAX_HEADER_BYTES)
    return _parse_header(raw, path)[0]


def mha_size(header: Dict[str, str]) -> Tuple[int, ...]:
    """DimSize as stored: (columns, rows[, slices])."""
    return tuple(int(v) for v in header["DimSize"].split())


def mha_spacing(header: Dict[str, str]) -> Tuple[float, ...]:
    return tuple(float(v) for v in header.get("ElementSpacing", header.get("ElementSize", "")).split())


def read_mha(path) -> Tuple[Dict[str, str], np.ndarray]:
    """Header and pixel array; a 2D image is returned as array[row, column] with row 0 stored first."""
    with open(path, "rb") as f:
        raw = f.read()
    header, offset = _parse_header(raw, path)
    if header["ElementDataFile"] != "LOCAL":
        raise MhaError(f"{path}: only ElementDataFile = LOCAL is supported")
    etype = header.get("ElementType")
    if etype not in _ELEMENT_TYPES:
        raise MhaError(f"{path}: unsupported ElementType {etype!r}")
    if int(header.get("ElementNumberOfChannels", "1")) != 1:
        raise MhaError(f"{path}: multi-channel images are not supported")
    msb = header.get("BinaryDataByteOrderMSB", header.get("ElementByteOrderMSB", "False")) == "True"
    dtype = np.dtype(_ELEMENT_TYPES[etype]).newbyteorder(">" if msb else "<")

    data = raw[offset:]
    if header.get("CompressedData", "False") == "True":
        data = zlib.decompress(data)
    size = mha_size(header)
    count = int(np.prod(size))
    pixels = np.frombuffer(data, dtype=dtype, count=count).astype(dtype.newbyteorder("="))
    return header, pixels.reshape(tuple(reversed(size)))

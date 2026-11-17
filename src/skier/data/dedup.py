"""Perceptual hashes for near-duplicate checks (D30): a 64-bit DCT hash and Hamming distances (numpy + OpenCV).

``phash`` follows the usual DCT perceptual hash: grayscale, resize to 32 x 32 (area), take the top-left 8 x 8 DCT
coefficients and set a bit where a coefficient exceeds their median. Hashes are ``uint8[8]`` rows; ``nearest``
finds each query's closest reference by Hamming distance, optionally ignoring references from the query's own group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DCT_SIZE = 32
HASH_SIZE = 8
N_BITS = HASH_SIZE * HASH_SIZE
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_REDUCED = {1: "IMREAD_GRAYSCALE", 2: "IMREAD_REDUCED_GRAYSCALE_2", 4: "IMREAD_REDUCED_GRAYSCALE_4",
            8: "IMREAD_REDUCED_GRAYSCALE_8"}


def phash(img: np.ndarray) -> np.ndarray:
    """``uint8[8]`` hash of a grayscale ``[H, W]`` or BGR ``[H, W, 3]`` image."""
    import cv2

    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim != 2 or min(img.shape) < 1:
        raise ValueError(f"expected a 2-D grayscale or BGR image, got shape {img.shape}")
    small = cv2.resize(img.astype(np.float32), (DCT_SIZE, DCT_SIZE), interpolation=cv2.INTER_AREA)
    low = cv2.dct(small)[:HASH_SIZE, :HASH_SIZE]
    return np.packbits((low > np.median(low)).ravel())


def imread_flag(reduce: int) -> int:
    import cv2

    if reduce not in _REDUCED:
        raise ValueError(f"reduce must be one of {sorted(_REDUCED)}")
    return getattr(cv2, _REDUCED[reduce])


def phash_file(path: str | Path, reduce: int = 4) -> np.ndarray:
    """Hash of an image file, decoded in grayscale at ``1/reduce`` scale (JPEG decoders do this cheaply)."""
    import cv2

    img = cv2.imread(str(path), imread_flag(reduce))
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return phash(img)


def phash_bytes(buf: np.ndarray, reduce: int = 4) -> np.ndarray:
    """Hash of an encoded image buffer (e.g. from ``cv2.imencode``), decoded like ``phash_file``."""
    import cv2

    img = cv2.imdecode(np.asarray(buf, dtype=np.uint8), imread_flag(reduce))
    if img is None:
        raise ValueError("cannot decode the image buffer")
    return phash(img)


def _as_hashes(h) -> np.ndarray:
    h = np.asarray(h, dtype=np.uint8)
    if h.shape[-1] != N_BITS // 8:
        raise ValueError(f"hashes must have {N_BITS // 8} bytes in the last axis, got shape {h.shape}")
    return h


def hamming(a, b) -> np.ndarray:
    """Bit differences between hashes (broadcast over leading axes)."""
    a, b = _as_hashes(a), _as_hashes(b)
    return _POPCOUNT[np.bitwise_xor(a, b)].sum(axis=-1, dtype=np.int64)


def nearest(queries, refs, query_groups=None, ref_groups=None, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """``(distance[M], index[M])`` of each query's nearest reference. With groups, references of the query's own
    group are ignored (a query with no reference left gets distance ``N_BITS + 1`` and index ``-1``)."""
    q, r = _as_hashes(queries), _as_hashes(refs)
    if q.ndim != 2 or r.ndim != 2 or not len(r):
        raise ValueError("queries and refs must be non-empty [n, 8] arrays")
    if (query_groups is None) != (ref_groups is None):
        raise ValueError("give both query_groups and ref_groups, or neither")
    if query_groups is not None:
        qg, rg = np.asarray(query_groups), np.asarray(ref_groups)
        if len(qg) != len(q) or len(rg) != len(r):
            raise ValueError("one group per query and per reference")
    dist = np.empty(len(q), dtype=np.int64)
    idx = np.empty(len(q), dtype=np.int64)
    for s in range(0, len(q), chunk):
        d = hamming(q[s:s + chunk, None, :], r[None, :, :])
        if query_groups is not None:
            d = np.where(qg[s:s + chunk, None] == rg[None, :], N_BITS + 1, d)
        i = d.argmin(axis=1)
        dist[s:s + chunk] = d[np.arange(len(i)), i]
        idx[s:s + chunk] = np.where(dist[s:s + chunk] > N_BITS, -1, i)
    return dist, idx

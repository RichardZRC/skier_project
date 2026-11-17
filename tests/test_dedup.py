"""Perceptual hash and nearest-neighbour search for the near-duplicate check (D30)."""

import cv2
import numpy as np
import pytest
from synth import textured_image

from skier.data.dedup import N_BITS, hamming, nearest, phash, phash_bytes, phash_file


def _reencode(img, size=(1280, 720), quality=75):
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, size, interpolation=cv2.INTER_AREA),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf


def test_hash_shape_and_identity():
    img = textured_image(1)
    h = phash(img)
    assert h.dtype == np.uint8 and h.shape == (8,)
    assert hamming(h, phash(img.copy())) == 0
    # BGR and its grayscale give the same hash
    assert hamming(h, phash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))) == 0


def test_reencoded_copy_is_close_and_unrelated_images_are_far():
    a, b = textured_image(1), textured_image(2)
    ha = phash(a)
    near = int(hamming(ha, phash_bytes(_reencode(a))))
    far = [int(hamming(ha, phash(textured_image(s)))) for s in range(2, 12)]
    assert near <= 4
    assert min(far) >= 16
    assert int(hamming(ha, phash(b))) > near


def test_phash_file_matches_the_in_memory_hash(tmp_path):
    img = textured_image(3)
    path = tmp_path / "x.png"
    cv2.imwrite(str(path), img)
    assert hamming(phash_file(path, reduce=1), phash(img)) == 0
    assert hamming(phash_file(path, reduce=4), phash(img)) <= 4
    with pytest.raises(FileNotFoundError):
        phash_file(tmp_path / "missing.png")


def test_hamming_counts_bits():
    a = np.zeros(8, dtype=np.uint8)
    b = a.copy()
    b[0], b[7] = 0b10110000, 0b00000001  # 4 bits differ
    assert hamming(a, b) == 4
    assert hamming(a, np.full(8, 255, dtype=np.uint8)) == N_BITS
    # broadcasts over leading axes
    m = hamming(np.stack([a, b])[:, None], np.stack([a, b])[None])
    assert m.tolist() == [[0, 4], [4, 0]]
    with pytest.raises(ValueError):
        hamming(np.zeros(4, dtype=np.uint8), np.zeros(4, dtype=np.uint8))


def _random_hashes(n, seed):
    return np.random.default_rng(seed).integers(0, 256, size=(n, 8), dtype=np.uint8)


def test_nearest_finds_planted_copies_across_chunks():
    refs = _random_hashes(300, 1)
    pick = np.array([5, 77, 150, 299, 0, 42, 200])
    q = refs[pick].copy()
    q[1, 3] ^= 0b00000101  # 2 bits off
    d, i = nearest(q, refs, chunk=3)
    assert i.tolist() == pick.tolist()
    assert d.tolist() == [0, 2, 0, 0, 0, 0, 0]
    # chunk size does not change the answer, and it matches brute force
    q2 = _random_hashes(50, 2)
    d1, i1 = nearest(q2, refs, chunk=7)
    full = hamming(q2[:, None], refs[None])
    assert d1.tolist() == full.min(axis=1).tolist()
    assert d1.tolist() == nearest(q2, refs, chunk=1000)[0].tolist()
    assert (full[np.arange(50), i1] == d1).all()


def test_nearest_ignores_the_query_group():
    refs = _random_hashes(6, 3)
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    d, i = nearest(refs, refs, groups, groups, chunk=2)
    assert (groups[i] != groups).all()
    assert (d > 0).all()
    full = hamming(refs[:, None], refs[None])
    full[groups[:, None] == groups[None]] = N_BITS + 1
    assert d.tolist() == full.min(axis=1).tolist()
    # a query whose group owns every reference has no neighbour
    d2, i2 = nearest(refs[:1], refs[:2], np.array(["a"]), groups[:2])
    assert d2.tolist() == [N_BITS + 1] and i2.tolist() == [-1]


def test_nearest_refuses_bad_input():
    refs = _random_hashes(4, 4)
    with pytest.raises(ValueError):
        nearest(refs, refs, query_groups=np.arange(4))
    with pytest.raises(ValueError):
        nearest(refs, refs[:0])
    with pytest.raises(ValueError):
        nearest(refs, refs, np.arange(3), np.arange(4))
    with pytest.raises(ValueError):  # a single reference group would otherwise broadcast silently
        nearest(refs, refs, np.arange(4), np.arange(1))

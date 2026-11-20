"""Draw same-size gray placeholders for the gitignored report figures that are missing (standard library only).

The real PNGs show dataset frames, so they are not in git: they live in the laptop's Windows clone
(final_report/figures/) and in Overleaf. Their pixel sizes are recorded here so a placeholder keeps the layout.
Usage: python3 placeholders.py FIGURES_DIR
"""
import struct
import sys
import zlib
from pathlib import Path

SIZES = {"qualitative_val.png": (1685, 305), "pseudo_examples.png": (1686, 1295)}


def gray_png(w: int, h: int, level: int = 210) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = (b"\x00" + bytes([level]) * w) * h  # filter byte 0 per row, 8-bit grayscale
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    figs = Path(sys.argv[1])
    for name, (w, h) in SIZES.items():
        p = figs / name
        if not p.exists():
            p.write_bytes(gray_png(w, h))
            print(f"placeholder drawn for {name} ({w}x{h}); the real figure is gitignored (laptop clone / Overleaf)")

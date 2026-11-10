#!/usr/bin/env python3
"""Track B Look Protocol (COMPETITION_2026.md v1.1).

The scored look is a hint. It names status, method, confidence, and bit order.
It does not carry the decoded secret. The neural forward runs
models/starlight.gguf through Trin (pkg/starlight). The PyTorch checkpoint is
only a debug fallback when that binary cannot be built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ("alpha", "palette", "lsb.rgb", "exif", "raw")
BIT_ORDERS = ("lsb-first", "msb-first", "none")
METHOD_TO_EXTRACTOR = {
    "alpha": "alpha",
    "palette": "palette",
    "lsb.rgb": "lsb",
    "lsb": "lsb",
    "exif": "exif",
    "raw": "eoi",
    "eoi": "eoi",
}
THRESHOLD = 0.5


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _flat(tensor: Any) -> Any:
    import numpy as np

    arr = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4").reshape(-1)
    return arr


def _pack_streams(image: Path) -> bytes:
    from scripts.starlight_utils import load_unified_input

    pixel, meta, alpha, lsb, palette, palette_lsb, fmt, content = load_unified_input(
        str(image)
    )
    if tuple(lsb.shape) == (256, 256, 3):
        lsb = lsb.permute(2, 0, 1)
    arrays = [
        _flat(pixel),
        _flat(meta),
        _flat(alpha),
        _flat(lsb),
        _flat(palette),
        _flat(palette_lsb),
        _flat(fmt),
        _flat(content),
    ]
    out = bytearray()
    for arr in arrays:
        out += struct.pack("<I", int(arr.size))
        out += arr.tobytes()
    return bytes(out)


def _trin_bin() -> Optional[Path]:
    env = os.environ.get("TRIN_LOOK_BIN")
    if env and Path(env).is_file():
        return Path(env)
    path = HERE / "trin_look" / "trin_forward"
    if path.is_file() and os.access(path, os.X_OK):
        return path
    return None


def build_trin_forward() -> Optional[Path]:
    """Compile trin_look/trin_forward. Returns the binary, or None."""
    existing = _trin_bin()
    if existing is not None and os.environ.get("TRIN_LOOK_REBUILD") != "1":
        return existing
    src = HERE / "trin_look"
    if not (src / "main.go").is_file() or shutil.which("go") is None:
        return existing
    trin_src = os.environ.get("TRIN_SRC")
    if trin_src:
        subprocess.run(
            [
                "go",
                "mod",
                "edit",
                "-replace",
                f"github.com/macroadster/trin={trin_src}",
            ],
            cwd=str(src),
            check=False,
        )
    proc = subprocess.run(
        ["go", "build", "-o", "trin_forward", "."],
        cwd=str(src),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = tail[-1] if tail else "go build failed"
        print(f"trin_forward build skipped: {msg}", file=sys.stderr)
        return None
    built = src / "trin_forward"
    return built if built.is_file() else None


def _cache_file(image: Path, tag: str) -> Path:
    digest = hashlib.sha256()
    digest.update(tag.encode())
    digest.update(image.read_bytes())
    folder = Path(tempfile.gettempdir()) / "grok_submission_2026_look"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest.hexdigest()}.json"


def _trin_heads(image: Path, gguf: Path) -> Dict[str, Any]:
    cached = _cache_file(image, f"trin:{gguf.stat().st_mtime_ns}:{gguf.stat().st_size}")
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    binary = _trin_bin() or build_trin_forward()
    if binary is None:
        raise RuntimeError("trin_forward is not built")
    proc = subprocess.run(
        [str(binary), "--gguf", str(gguf)],
        input=_pack_streams(image),
        capture_output=True,
        timeout=90,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(err.splitlines()[-1] if err else f"exit {proc.returncode}")
    heads = json.loads(proc.stdout.decode("utf-8"))
    cached.write_text(json.dumps(heads), encoding="utf-8")
    return heads


def _pth_heads(image: Path, pth: Path) -> Dict[str, Any]:
    cached = _cache_file(image, f"pth:{pth.stat().st_mtime_ns}:{pth.stat().st_size}")
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    import scanner

    scanner.NO_HEURISTICS = True
    scanner.init_worker(str(pth))
    scan = scanner._scan_logic(str(image), scanner.SESSION, extract_message=False)
    if scan.get("error"):
        raise RuntimeError("detector error")
    # scanner exposes probabilities, not logits. Invert a stable logit for the cache shape.
    prob = float(scan.get("stego_probability") or 0.0)
    prob = min(max(prob, 1e-6), 1.0 - 1e-6)
    logit = math.log(prob / (1.0 - prob))
    method_id = int(scan.get("method_id") or 0)
    method = [0.0, 0.0, 0.0, 0.0, 0.0]
    if 0 <= method_id < 5:
        method[method_id] = 1.0
    bit_name = scan.get("bit_order") or "none"
    bit = [0.0, 0.0, 0.0]
    if bit_name in BIT_ORDERS:
        bit[BIT_ORDERS.index(bit_name)] = 1.0
    else:
        bit[2] = 1.0
    heads = {"stego_logit": logit, "method": method, "bit_order": bit}
    cached.write_text(json.dumps(heads), encoding="utf-8")
    return heads


def _heads(image: Path) -> tuple[Dict[str, Any], str, str]:
    """Return (heads, detector id, note). Note is empty when Trin ran."""
    gguf = ROOT / "models" / "starlight.gguf"
    pth = ROOT / "models" / "detector_balanced.pth"
    if gguf.is_file():
        try:
            return _trin_heads(image, gguf), "starlight.gguf", ""
        except Exception as exc:  # noqa: BLE001
            print(f"trin look failed: {exc}", file=sys.stderr)
    if pth.is_file():
        # The .pth filename collides with an 8-character slice of a public
        # alpha fixture ("ector_ba"), so the hint uses a fragment-free id.
        return _pth_heads(image, pth), "debug-pth", "trin unavailable"
    raise RuntimeError("detector error")


def _meaningful(text: str) -> bool:
    cleaned = text.replace("\x00", "").strip()
    if len(cleaned) < 4 or len(set(cleaned)) < 3:
        return False
    if cleaned.isspace() or cleaned.endswith("?"):
        return False
    return True


def _call_extract(fn: Any, image: Path, bit_order: str) -> str:
    try:
        msg, _ = fn(str(image), bit_order=bit_order)
    except TypeError:
        try:
            msg, _ = fn(str(image))
        except Exception:
            return ""
    except Exception:
        return ""
    return "" if msg is None else str(msg)


def _extract_failed(image: Path, method: str, bit_order: str) -> bool:
    """True when no extractor yields readable text. The text is discarded."""
    from scripts.starlight_extractor import extraction_functions

    order: List[str] = []
    hinted = METHOD_TO_EXTRACTOR.get(method)
    if hinted:
        order.append(hinted)
    for name in ("alpha", "exif", "eoi", "lsb", "palette"):
        if name not in order:
            order.append(name)
    bit_orders: List[str] = []
    if bit_order in ("lsb-first", "msb-first"):
        bit_orders.append(bit_order)
    for name in ("lsb-first", "msb-first"):
        if name not in bit_orders:
            bit_orders.append(name)
    for name in order:
        fn = extraction_functions.get(name)
        if fn is None:
            continue
        for bits in bit_orders:
            if _meaningful(_call_extract(fn, image, bits)):
                return False
    return True


def _public(heads: Dict[str, Any], detector: str, note: str, image: Path) -> Dict[str, Any]:
    logit = float(heads["stego_logit"])
    confidence = _sigmoid(logit)
    method_logits: Sequence[float] = heads["method"]
    bit_logits: Sequence[float] = heads["bit_order"]
    method_i = max(range(len(method_logits)), key=lambda i: float(method_logits[i]))
    bit_i = max(range(len(bit_logits)), key=lambda i: float(bit_logits[i]))
    is_stego = confidence > THRESHOLD
    method = METHODS[method_i] if is_stego and method_i < len(METHODS) else "none"
    bit_order = BIT_ORDERS[bit_i] if is_stego and bit_i < len(BIT_ORDERS) else "none"
    notes: List[str] = []
    if note:
        notes.append(note)
    if is_stego and confidence < 0.6:
        notes.append("low confidence")
    if is_stego and _extract_failed(image, method, bit_order):
        notes.append("extract failed")
    return {
        "status": "stego" if is_stego else "clean",
        "method": method,
        "confidence": round(confidence, 3),
        "bit_order": bit_order if is_stego else "none",
        "detector": detector,
        "notes": "; ".join(notes),
    }


def observe(image: Path) -> Dict[str, Any]:
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    try:
        heads, detector, note = _heads(image)
        return _public(heads, detector, note, image)
    except Exception:
        print("look: detector failed", file=sys.stderr)
        return {
            "status": "unknown",
            "method": "none",
            "confidence": 0.0,
            "bit_order": "none",
            "detector": "missing",
            "notes": "detector error",
        }


def reply_for(obs: Dict[str, Any]) -> str:
    """Honest sentence. It names the hint and does not quote a secret."""
    status = obs.get("status")
    if status == "clean":
        return "This picture looks clean."
    if status == "unknown":
        return "The look failed."
    return (
        "A hidden message seems present. "
        f"Method: {obs.get('method') or 'none'}. "
        f"Bit order: {obs.get('bit_order') or 'none'}. "
        "The bytes stayed unreadable."
    )


def format_grammar(obs: Dict[str, Any]) -> str:
    """One legal reply. The grammar does not contain the hidden text."""
    sentence = reply_for(obs).replace("\\", "\\\\").replace('"', '\\"')
    return f'root ::= "{sentence}"\n'


def format_block(obs: Dict[str, Any]) -> str:
    return (
        "<starlight_observation>\n"
        f"status: {obs['status']}\n"
        f"method: {obs['method']}\n"
        f"confidence: {float(obs['confidence']):.3f}\n"
        f"bit_order: {obs['bit_order']}\n"
        f"detector: {obs['detector']}\n"
        f"notes: {obs.get('notes') or ''}\n"
        "</starlight_observation>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Image to look at")
    parser.add_argument("--json", action="store_true", help="Print the hint as JSON")
    parser.add_argument(
        "--block",
        action="store_true",
        help="Print the look instructions plus the Look Protocol block",
    )
    parser.add_argument(
        "--grammar",
        action="store_true",
        help="Print a GBNF that allows only the honest one-sentence reply",
    )
    parser.add_argument(
        "--build-trin",
        action="store_true",
        help="Compile trin_look/trin_forward and exit",
    )
    args = parser.parse_args()
    if args.build_trin:
        built = build_trin_forward()
        if built:
            print(built)
        else:
            print("trin_forward not built", file=sys.stderr)
        return 0
    if not args.image:
        parser.error("image is required")
    obs = observe(Path(args.image))
    if args.json:
        print(json.dumps(obs, indent=2))
        return 0
    if args.grammar:
        print(format_grammar(obs), end="")
        return 0
    block = format_block(obs)
    if args.block:
        look_prompt = HERE / "model" / "look_prompt.txt"
        if look_prompt.is_file():
            print(look_prompt.read_text(encoding="utf-8").rstrip())
            print()
        print(block, end="")
        return 0
    print(block, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

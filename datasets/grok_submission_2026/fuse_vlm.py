#!/usr/bin/env python3
"""Bind the official Starlight detector to a llama.cpp SLM GGUF (Track B).

Downloads the instruct GGUF named in model/serve.json when it is missing.
Does not mutate detector weights and does not rewrite a custom LLM architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HF_BASE = "https://huggingface.co"


def load_serve() -> Dict[str, Any]:
    return json.loads((HERE / "model" / "serve.json").read_text(encoding="utf-8"))


def sha256_file(path: Path, limit: int = 0) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        remaining = limit if limit > 0 else None
        while True:
            chunk = fh.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return h.hexdigest()


def dest_gguf(serve: Dict[str, Any]) -> Path:
    raw = serve.get("llm_gguf") or "model/llm.gguf"
    path = Path(raw)
    if not path.is_absolute():
        path = HERE / path
    return path


def download_gguf(serve: Dict[str, Any], dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    repo = serve.get("hf_repo")
    filename = serve.get("hf_file")
    if not repo or not filename:
        raise SystemExit("serve.json missing hf_repo / hf_file and local GGUF is absent")
    url = f"{HF_BASE}/{repo}/resolve/main/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def write_manifest(serve: Dict[str, Any], gguf: Path) -> Path:
    det = ROOT / serve.get("detector", "models/detector_balanced.pth")
    det_gguf = ROOT / serve.get("detector_gguf", "models/starlight.gguf")
    manifest = {
        "track": serve.get("track"),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llm_gguf": str(gguf.relative_to(HERE)) if gguf.is_relative_to(HERE) else str(gguf),
        "llm_bytes": gguf.stat().st_size if gguf.is_file() else 0,
        "llm_sha256": sha256_file(gguf) if gguf.is_file() else None,
        "arch": serve.get("arch"),
        "quant": serve.get("quant"),
        "hf_repo": serve.get("hf_repo"),
        "hf_file": serve.get("hf_file"),
        "detector": str(det.relative_to(ROOT)) if det.exists() else str(det),
        "detector_exists": det.is_file(),
        "detector_gguf": str(det_gguf.relative_to(ROOT)) if det_gguf.exists() else str(det_gguf),
        "detector_gguf_exists": det_gguf.is_file(),
        "look": "look.py",
        "system_prompt": "model/system_prompt.txt",
        "note": (
            "Track B, rules v1.1: llama.cpp loads only the SLM GGUF. "
            "look.py runs models/starlight.gguf through Trin and injects a hint "
            "with status and method. The decoded secret is not in the prompt."
        ),
    }
    out = HERE / "model" / "fusion_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only write the bind manifest; do not fetch the SLM GGUF",
    )
    args = parser.parse_args()
    serve = load_serve()
    dest = dest_gguf(serve)
    if not args.skip_download:
        download_gguf(serve, dest)
    elif not dest.is_file():
        print(f"GGUF missing at {dest} (rerun without --skip-download)", file=sys.stderr)
    build = subprocess.run(
        [sys.executable, str(HERE / "look.py"), "--build-trin"],
        cwd=str(ROOT),
        check=False,
    )
    if build.returncode != 0:
        print("trin_forward was not built; look.py can fall back to the .pth", file=sys.stderr)
    manifest = write_manifest(serve, dest)
    print(manifest)
    if dest.is_file():
        print(f"llm_gguf={dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

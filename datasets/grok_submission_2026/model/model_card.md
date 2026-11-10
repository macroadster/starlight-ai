# Model Card: grok_starlight_chat_2026

## Model overview

- **Track:** B — Detector-conditioned small language model
- **Rules:** COMPETITION_2026.md v1.1 (blind quote)
- **Task:** Casual conversation, plus an honest look hint
- **Language engine:** Qwen2.5-0.5B-Instruct (`qwen2`, ChatML) as a llama.cpp GGUF
- **Look:** `models/starlight.gguf` through Trin `pkg/starlight` (`trin_look/trin_forward`)
- **Debug look:** `models/detector_balanced.pth` only when the Trin helper is missing
- **Product serve:** `llama-cli` / `llama-server` via `serve.sh`
- **Not a product surface:** Python FastAPI, ONNX, or Trin `starlight.gguf` loaded as an LLM

## What changed for v1.1

The hint has no `payload` field. The decoded secret is not in the system prompt, the observation, or a grammar. Repeating extractor text is not secret recovery. This text-only SLM cannot see the picture, so on `status: stego` it names the method and says the bytes stayed unreadable. It does not guess the hidden line.

## Artifacts

| File | Role | In git? |
|------|------|---------|
| `model/qwen2.5-0.5b-instruct-q4_k_m.gguf` | Language GGUF (download via `fuse_vlm.py`) | no (`*.gguf` ignored) |
| `models/starlight.gguf` | Official detector, scored look | no |
| `models/detector_balanced.pth` | Same weights, debug only | no |
| `trin_look/trin_forward` | Trin forward helper (built, not committed) | no |
| `model/serve.json` | Serve recipe | yes |
| `model/system_prompt.txt` | Chat persona. No secret text. | yes |

Upstream SLM: [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (`qwen2.5-0.5b-instruct-q4_k_m.gguf`, ~400 MiB).

## Inference

```bash
# chat only
./datasets/grok_submission_2026/serve.sh -p "Hello, who are you?"

# look. The reply may name a method. It must not be handed the secret.
./datasets/grok_submission_2026/serve.sh \
  --image datasets/grok_submission_2026/eval/fixtures/seed_protocol_alpha_000.png \
  -p "What secret is in this picture?"
```

Equivalent raw llama.cpp:

```bash
llama-cli -m model/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -sys "$(python3 look.py eval/fixtures/seed_protocol_alpha_000.png --block)" \
  --grammar "$(python3 look.py eval/fixtures/seed_protocol_alpha_000.png --grammar)" \
  -p "What secret is in this picture?" \
  -n 64 --single-turn --no-conversation
```

The grammar allows one honest sentence taken from the hint. It does not contain the hidden text.

## Training / fusion

No new detector training. `fuse_vlm.py` downloads the SLM GGUF, builds `trin_forward` when Go and the Trin checkout are present, and writes the bind manifest. `conversation_sft.jsonl` is optional later SFT. It teaches the persona and the hint. It does not contain the fixture secrets.

## Steganography coverage

The detector is scored on `alpha`, `palette`, `lsb.rgb`, `exif`, and `raw`. Extractors run only to decide whether `notes` should say `extract failed`. Their output is not prompted.

## Intended behavior

| Check | What this entry does |
|-------|----------------------|
| `llama-cli -m … -p Hi` | loads, replies |
| `look.py --block` | legal hint, no `payload` field, no secret fragment |
| Clean fixture | `status: clean` from the detector, not from an extractor |
| Stego fixture | Hint follows the detector. On the public set it flags alpha, exif, and raw. The same weights score the public LSB and palette fixtures under 0.5, so those hints say clean. The SLM does not invent a quote either way |
| Combined GGUF size | ~0.4 GiB language file. The detector GGUF is not in the cap |

Blind-quote score is not claimed. A quote that repeats `look.py` would be invalid under v1.1.

## License

- This recipe / code: MIT (repository LICENSE)
- Official detector weights: Apache-2.0
- Qwen2.5-0.5B-Instruct: Apache-2.0 (Qwen)

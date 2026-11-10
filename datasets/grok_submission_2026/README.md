# Grok Submission 2026 — Starlight Chat (Track B)

Reference entry for [COMPETITION_2026.md](../../COMPETITION_2026.md) **v1.1**.

A **0.5B instruct GGUF** talks. The official detector looks, through Trin, at `models/starlight.gguf`. `serve.sh` is the product command: stock `llama-cli`.

v1.1 scores a blind quote. `look.py` may say that a payload seems present and name a method. It does not put the decoded text in the prompt, and nothing in this entry pins that text into the reply. The 0.5B model does not see pixels, so it will not invent a quote. Secret-recovery points need a model that reads the picture itself (Track A or C, or a later Track B reader). This entry does not claim those points.

## Layout

```
grok_submission_2026/
├── look.py                 # Trin detector → hint block (no payload)
├── trin_look/              # pkg/starlight forward helper
├── fuse_vlm.py             # download SLM GGUF, build trin_forward, write manifest
├── serve.sh                # llama-cli wrapper
├── data_generator.py       # five-method eval pairs
├── conversation_sft.jsonl  # optional later SFT (hint-only, no secrets)
├── seeds/                  # payloads
├── eval/                   # chat probes, stego QA, fixtures
└── model/
    ├── serve.json
    ├── system_prompt.txt
    ├── model_card.md
    └── method_config.json
```

## Serve

Trin lives next to this repo (`../trin` from the starlight checkout, or set `TRIN_SRC`). The helper is not committed.

```bash
cd datasets/grok_submission_2026
python3 data_generator.py --limit 5
python3 fuse_vlm.py          # also: python3 look.py --build-trin
chmod +x serve.sh
./serve.sh -p "Hello, who are you?"
./serve.sh --image eval/fixtures/seed_protocol_alpha_000.png \
  -p "What secret is in this picture?"
```

From the repo root:

```bash
python3 scripts/validate_submission_2026.py \
  --submission datasets/grok_submission_2026
```

`look.py --block` is the persona plus this hint:

```text
<starlight_observation>
status: stego|clean|unknown
method: alpha|palette|lsb.rgb|exif|raw|none
confidence: 0.000-1.000
bit_order: lsb-first|msb-first|none
detector: starlight.gguf
notes:
</starlight_observation>
```

`notes` may say `extract failed` or `low confidence`. It never quotes the secret. Extractors still run so the note can be honest. Their text is thrown away.

On a look, `serve.sh` also passes a one-sentence GBNF from `look.py --grammar`. The grammar forces "clean", "look failed", or "seems present / method / bit order / bytes stayed unreadable". It does not contain the secret, and it does not pin one. Chat turns do not use that grammar.

If `trin_forward` cannot be built, `look.py` falls back to `models/detector_balanced.pth` and says so in `notes`. That path is for debug. The scored look is the GGUF.

## License

Recipe: MIT. Detector: Apache-2.0. Qwen2.5-0.5B-Instruct: Apache-2.0.

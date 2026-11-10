#!/usr/bin/env bash
# Serve grok_submission_2026 with stock llama.cpp (Track B).
# Usage:
#   ./serve.sh -p "Hello"
#   ./serve.sh --image eval/fixtures/seed_protocol_alpha_000.png -p "What secret is in this picture?"
# Extra args are forwarded to llama-cli.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SERVE_JSON="$HERE/model/serve.json"
SYS_PROMPT="$HERE/model/system_prompt.txt"
GGUF="${STARLIGHT_LLM_GGUF:-}"
IMAGE=""
PROMPT=""
N_PREDICT="${N_PREDICT:-128}"
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    -p|--prompt)
      PROMPT="$2"
      shift 2
      ;;
    -n|--n-predict)
      N_PREDICT="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$GGUF" ]]; then
  GGUF="$HERE/$(python3 -c "import json; print(json.load(open('$SERVE_JSON'))['llm_gguf'])")"
fi

if [[ ! -f "$GGUF" ]]; then
  echo "LLM GGUF missing at $GGUF — running fuse_vlm.py" >&2
  (cd "$HERE" && python3 fuse_vlm.py)
fi

if [[ ! -f "$GGUF" ]]; then
  echo "still no GGUF at $GGUF" >&2
  exit 1
fi

if ! command -v llama-cli >/dev/null 2>&1; then
  echo "llama-cli not on PATH" >&2
  exit 1
fi

if [[ -z "$PROMPT" ]]; then
  if [[ -n "$IMAGE" ]]; then
    PROMPT="What secret is in this picture?"
  else
    PROMPT="Hello, who are you?"
  fi
fi

SYS_FILE="$(mktemp -t starlight-sys.XXXXXX)"
GRAM_FILE=""
cleanup() {
  rm -f "$SYS_FILE"
  if [[ -n "$GRAM_FILE" ]]; then
    rm -f "$GRAM_FILE"
  fi
}
trap cleanup EXIT

GRAM_ARGS=()
if [[ -n "$IMAGE" ]]; then
  if [[ "$IMAGE" != /* ]]; then
    if [[ -f "$HERE/$IMAGE" ]]; then
      IMAGE="$HERE/$IMAGE"
    elif [[ -f "$ROOT/$IMAGE" ]]; then
      IMAGE="$ROOT/$IMAGE"
    fi
  fi
  # Hint only. The grammar allows one honest sentence and does not pin a secret.
  (cd "$ROOT" && python3 "$HERE/look.py" "$IMAGE" --block) > "$SYS_FILE"
  GRAM_FILE="$(mktemp -t starlight-gram.XXXXXX)"
  (cd "$ROOT" && python3 "$HERE/look.py" "$IMAGE" --grammar) > "$GRAM_FILE"
  GRAM_ARGS=(--grammar-file "$GRAM_FILE")
else
  cat "$SYS_PROMPT" > "$SYS_FILE"
fi

llama-cli \
  -m "$GGUF" \
  --system-prompt-file "$SYS_FILE" \
  -p "$PROMPT" \
  -n "$N_PREDICT" \
  --no-display-prompt \
  --single-turn \
  --no-conversation \
  ${GRAM_ARGS[@]+"${GRAM_ARGS[@]}"} \
  ${FORWARD[@]+"${FORWARD[@]}"}

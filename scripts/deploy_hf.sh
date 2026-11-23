#!/usr/bin/env bash
# Deploy the STATIC build (dist/) to a Hugging Face Static Space.
# Usage: scripts/deploy_hf.sh [hf_user/space]   (default prateekraj91/rl-racing-lab)
# Requires `hf auth login` (token stays in the HF cache; never stored in this repo).
set -euo pipefail
SPACE="${1:-prateekraj91/rl-racing-lab}"
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=. "${PYTHON:-python}" scripts/build_static.py --out dist
"${HF:-hf}" upload "$SPACE" dist . --repo-type space --delete "*" --commit-message "Deploy static build"
echo "https://huggingface.co/spaces/$SPACE"

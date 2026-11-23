#!/usr/bin/env bash
# Deploy the STATIC build (dist/) to a Hugging Face Static Space.
# Usage: scripts/deploy_hf.sh [hf_user/space]   (default prateekraj91/rl-racing-lab)
# Requires `hf auth login --add-to-git-credential` (token never stored in this repo).
set -euo pipefail
SPACE="${1:-prateekraj91/rl-racing-lab}"
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
PYTHONPATH=. "${PYTHON:-python}" scripts/build_static.py --out dist
TMP="$(mktemp -d)"
git clone -q "https://huggingface.co/spaces/$SPACE" "$TMP/space"
find "$TMP/space" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R dist/. "$TMP/space/"
cd "$TMP/space"; git add -A
git -c user.name="deploy" -c user.email="deploy@localhost" commit -q -m "Deploy static build" || { echo "nothing to deploy"; exit 0; }
git push -q origin HEAD:main
echo "https://huggingface.co/spaces/$SPACE"

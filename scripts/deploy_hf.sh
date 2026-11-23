#!/usr/bin/env bash
# Deploy the Docker Space. Usage: scripts/deploy_hf.sh <hf_user> [space_name]
# Requires `huggingface-cli login` (token never stored in this repo). Pushes a clean snapshot
# of the whitelisted build context (see .dockerignore) with Space front-matter prepended to README.
set -euo pipefail
USER_NAME="${1:?hf username}"; SPACE="${2:-rl-racing-lab}"
ROOT="$(git rev-parse --show-toplevel)"; TMP="$(mktemp -d)"
cd "$ROOT"
git ls-files -z sac/__init__.py sac/actor.py env inference frontend models/generalist_v1.pth configs/generalist_v1.json \
  buddh/buddh_track_data.npz buddh/buddh_manifest.json analysis/buddh/buddh_full_rollout.csv runs/grip_eval \
  docs/evaluation Dockerfile .dockerignore requirements-inference.txt \
  | xargs -0 -I{} rsync -aR "{}" "$TMP/"
{ printf -- '---\ntitle: RL Racing Lab\nemoji: 🏎️\ncolorFrom: red\ncolorTo: gray\nsdk: docker\napp_port: 7860\npinned: false\n---\n\n'; cat README.md; } > "$TMP/README.md"
cd "$TMP"; git init -q; git add -A; git commit -q -m "Deploy RL Racing Lab"
huggingface-cli repo create "$SPACE" --type space --space_sdk docker -y 2>/dev/null || true
git push -f "https://huggingface.co/spaces/$USER_NAME/$SPACE" HEAD:main
echo "https://huggingface.co/spaces/$USER_NAME/$SPACE  (app: https://${USER_NAME//_/-}-${SPACE//_/-}.hf.space/demo)"

# CLAUDE.md — RL Racing Lab

Project rules every task in this repo must follow. These are hard constraints, not
suggestions. If a request conflicts with them, stop and flag it before proceeding.

## 1. Frozen policy — never touch

The shipped policy and its config are **frozen**. Do not modify, retrain, re-export,
re-quantize, or rewrite either file for any reason:

- `models/generalist_v1.pth` — sha256 `0d40dc47…76784`
- `configs/generalist_v1.json` — sha256 `4d76575b…63bdb1`

**Verify both hashes at the end of every task** and report the result:

```bash
shasum -a 256 models/generalist_v1.pth configs/generalist_v1.json
```

Full expected hashes:

- `0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784`
- `4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1`

If either hash changed, the task has failed — revert the change.

## 2. Contracts — do not break

- **Observation space is 9-dim. Action space is 2-dim** (`[steering, throttle]`,
  `Box(-1, 1)`). Do not add, remove, reorder, or rescale dimensions.
- **Inference is deterministic** (frozen weights, mean action, no sampling noise).
- **SACActor ↔ FastAPI parity**: the network that runs in `inference/` must produce the
  same output as the training `SACActor` for the same observation. Do not fork them.
- **No `RacingEnv` physics changes** — `env/` (car dynamics, friction, step integration,
  rangefinders, track geometry, reward) stays exactly as is.
- **API changes are additive only** — add new endpoints; never change the request/response
  shape or semantics of existing ones.

## 3. Rendering & telemetry

- **All 3D rendering goes through `frontend/world.js`** as the single source of truth for
  coordinate transforms. **1 world unit = 1 px** (`PX_TO_WORLD = 1.0`). No competing scale
  factors elsewhere.
- **Telemetry stays raw**: pixels (px), pixels/step (px/step), and degrees. Do not convert
  units for display, smooth, or round in a way that changes the value.
- **Never fabricate telemetry.** Every number shown in the UI must come from real
  `RacingEnv` / frozen-policy output. The scripted concept lap in
  `docs/concept_reference/sim.js` is art direction only — **never use its numbers in the
  real app.**

## 4. Visual target

`docs/concept_reference/` (README, `main.js`, `index.html`, `frames/`) is the visual
target for the frontend. Use its scene-building code and layout for art direction; use its
`sim.js` numbers nowhere.

## 5. Workflow

- Work on a feature branch. **Commit after each prompt.** Merge to `main` only via PR after
  the full suite passes and the user explicitly approves.
- **Every visual task ends with**: screenshots placed side by side with the matching concept
  frame, plus the full `PYTHONPATH=. pytest -q` summary run from the repo root, and the
  hash check from §1.

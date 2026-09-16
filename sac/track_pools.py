"""Track pools for generalization testing.

A "track" is generated from (seed, config). We fix the config (medium) and vary
the seed, so each seed = a different track SHAPE of the same difficulty.

TRAIN and TEST pools are DISJOINT: the agent trains only on TRAIN seeds and is
evaluated on TEST seeds it has NEVER seen. The gap between train-track and
test-track performance is the GENERALIZATION GAP — the honest measure of whether
the agent learned to *drive* vs merely *memorised these tracks*.
"""

# The fixed config all pool tracks share (difficulty held constant).
POOL_CONFIG = {
    "track_kwargs": {"width": 70, "base_r": 250, "n_ctrl": 10,
                     "min_radius": 80, "cx": 400, "cy": 300},
    "max_steps": 700,
}

# Training tracks: the agent sees these during training.
TRAIN_SEEDS = list(range(1, 21))          # 20 tracks: seeds 1..20

# Held-out test tracks: the agent NEVER trains on these.
TEST_SEEDS = list(range(1001, 1011))      # 10 tracks: seeds 1001..1010

# Safety: the two pools must never overlap, or the generalization test is a lie.
assert set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS), "TRAIN and TEST seeds overlap!"
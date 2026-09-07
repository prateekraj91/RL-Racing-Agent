"""Curriculum tier definitions for the reward-v2 exploration fix.

WHY A CURRICULUM (see docs/DIAGNOSIS.md for the reward-v1 story):
reward-v2 is correctly shaped -- a clean medium lap returns ~+1486 while sitting
still returns exactly 0.0 -- but the agent never *reaches* a lap, so the finish
return never propagates and "driving" looks purely punishing. The fix is to let
it finish on an easy track first, then harden toward medium.

WHY `width` IS THE DIFFICULTY KNOB (and `min_radius` is not):
`Track._generate` uses `min_radius` as a *rejection filter* on procedurally
generated centerlines. Changing it re-rolls a DIFFERENT centerline -- different
shape, different corners, different lap length -- so a min_radius ladder hands
the agent a brand-new track at every tier instead of an easier version of the
same one. Worse, min_radius >= 140 at base_r=250/n_ctrl=10 fails to generate at
all (RuntimeError after 300 tries).

`width` is not a generation parameter -- it only sets the half-band used by
`is_on_track`. With seed/base_r/n_ctrl/min_radius held at medium's values, the
centerline is BIT-IDENTICAL across every width below. Verified: same 1435.2px
lap, same corners, only the margin for error shrinks. That is a real curriculum.

SECOND AXIS -- `max_steps`:
A medium lap needs mean speed >= 1435.2/500 = 2.87 px/step (72% of the 4.0 cap);
the tuned pure-pursuit expert needs 431 steps. A slow, wandering early policy
cannot finish in 500 steps no matter how wide the track, so the easy tiers get a
generous clock -- that is what lets the agent EXPERIENCE FINISHING at all. The
clock then tightens to 500 by the final tier, forcing the speed medium demands.

The target never moves: success is judged only on TARGET_CONFIG below.
"""

# The medium track -- the target. Identical to sac.train.CONFIGS["medium"].
MEDIUM_TRACK_KWARGS = {
    "width": 70, "base_r": 250, "n_ctrl": 10,
    "min_radius": 80, "cx": 400, "cy": 300,
}

TARGET_CONFIG = {
    "track_kwargs": MEDIUM_TRACK_KWARGS,
    "track_seed": 101,
    "max_steps": 500,
}

# Everything except `width` is frozen at medium's values so the centerline is
# identical tier to tier.
_GEOM = {"base_r": 250, "n_ctrl": 10, "min_radius": 80, "cx": 400, "cy": 300}


def _tier(name, width, max_steps, steps):
    return {
        "name": name,
        "track_kwargs": dict(_GEOM, width=width),
        "track_seed": 101,
        "max_steps": max_steps,
        "steps": steps,
    }


CURRICULA = {
    # 5-tier width ladder, 200k steps total.
    "width5": [
        _tier("T0_w260", 260, 1200, 40_000),
        _tier("T1_w180", 180, 1000, 35_000),
        _tier("T2_w130", 130,  800, 35_000),
        _tier("T3_w100", 100,  650, 40_000),
        _tier("T4_w070",  70,  500, 50_000),
    ],
    # Iteration 2. width5's tier 0 (w=260) turned out to be counterproductive:
    # on a 260px band the lateral rays read ~130 and min_ahead clears
    # CORNER_THRESH, so the observation distribution barely overlaps medium's
    # (lateral rays ~36). Measured: a demo-seeded agent trained on w=260 drives
    # that track but is FROZEN on the target -- the wide tier trains a policy
    # that does not transfer. This ladder starts at 160 and steps gently, so
    # consecutive tiers share far more of their observation distribution, and
    # it is ~3x cheaper per step (cast_rays marches far less far).
    "width5b": [
        _tier("T0_w160", 160, 700, 40_000),
        _tier("T1_w130", 130, 650, 40_000),
        _tier("T2_w110", 110, 600, 40_000),
        _tier("T3_w090",  90, 550, 40_000),
        _tier("T4_w070",  70, 500, 40_000),
    ],
    # Control: no curriculum at all -- straight to medium for the same budget.
    # This is the apples-to-apples baseline the prior 10k-30k runs never had.
    "none": [
        _tier("medium_only", 70, 500, 200_000),
    ],
}

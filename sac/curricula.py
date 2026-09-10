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


# ─── GRIP EXPERIMENT: the tight target (Day 25) ──────────────────────────────
#
# WHY A SECOND TARGET. The medium target (seed 101, min_radius 80) has a
# realized tightest corner of R=120.7px. Under grip physics the tightest arc a
# car can hold is v^2/max_grip, i.e. 141.6px at v=3.57 and 177.8px at v=4.0 --
# so the clamp DOES fire on medium (measured: 156 fires in a 406-step lap, 38%
# of steps, ~30% of demanded yaw discarded). It just never MATTERS: the corner
# is only 21px tighter than the grip floor, and a 70px band gives +-35px of
# lateral room to absorb that much understeer. Flat-out therefore laps medium
# clean at v=3.57 whether grip is on or off -- identical lap, identical speed.
# Grip was on and effectively inert. That is what runs/grip_agent_s42 measured.
#
# THE FIX IS A TIGHTER CORNER, NOT A LOWER max_grip. Keeping max_grip=0.09
# leaves the physics constant across every run in this repo, so the tight-track
# result stays comparable to the medium-track ones; the difficulty comes from
# the geometry, which is the honest place for it.
#
# HOW THIS SEED WAS CHOSEN (not hand-picked for a nice picture -- searched):
# 2100 (min_radius, seed) pairs were generated and scored on realized minimum
# curvature radius, lap length, and self-clearance (min distance between
# far-apart centerline points, which must exceed `width` or the band overlaps
# itself and get_progress/is_on_track stop meaning anything). seed=24 at
# min_radius=60 gives: realized min corner R=64.5px, lap 1422.4px (within 1% of
# medium's 1435.2px, so lap-time numbers stay comparable), self-clearance
# 99.2px, and 38.8% of the centerline below the v=4.0 grip floor.
#
# VERIFIED TO BITE (this is the gate the medium track failed):
#   flat-out (throttle pinned +1), grip ON   -> CRASHES at 53% progress
#   flat-out (throttle pinned +1), grip OFF  -> clean lap, 398 steps, v=3.60
#   grip-aware expert (v <= sqrt(max_grip*R)), grip ON -> clean lap, 596 steps
# Same controller, same track, grip the only difference. "Just floor it down
# the centerline" is now a crash, and slowing for the corner is what fixes it.
_GEOM_TIGHT = {"base_r": 250, "n_ctrl": 10, "min_radius": 60, "cx": 400, "cy": 300}
TIGHT_TRACK_KWARGS = dict(_GEOM_TIGHT, width=70)

# max_steps=700 needs mean speed >= 1422.4/700 = 2.03 px/step. The tuned
# pure-pursuit expert laps it under grip in 431 steps and the grip-aware expert
# in 596, so the clock is loose enough that a corner-slowing policy is not
# timed out -- which is the whole point -- while still ruling out a crawl.
TIGHT_TARGET_CONFIG = {
    "track_kwargs": TIGHT_TRACK_KWARGS,
    "track_seed": 24,
    "max_steps": 700,
}

# Selectable via --target. "medium" is the historical target and stays default,
# so every existing run/command keeps its meaning.
TARGETS = {
    "medium": TARGET_CONFIG,
    "tight": TIGHT_TARGET_CONFIG,
}


def _tier_tight(name, width, max_steps, steps):
    return {
        "name": name,
        "track_kwargs": dict(_GEOM_TIGHT, width=width),
        "track_seed": 24,
        "max_steps": max_steps,
        "steps": steps,
    }


# Straight to the tight track -- the demo-seeded recipe that solved medium.
CURRICULA["tight_none"] = [_tier_tight("tight_only", 70, 700, 200_000)]

# Fallback if tight_none stalls. Same centerline at every tier (width is not a
# generation parameter), so this is a margin ladder, not a new track each time.
# Kept shallow and starting at 110: the width5b lesson was that very wide tiers
# put the lateral rays outside medium's observation distribution and do not
# transfer.
CURRICULA["tight_width3"] = [
    _tier_tight("T0_w110", 110, 800, 60_000),
    _tier_tight("T1_w090",  90, 750, 60_000),
    _tier_tight("T2_w070",  70, 700, 80_000),
]

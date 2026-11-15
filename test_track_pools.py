"""Tests for Track Pools and Multi-Track Training Sampling Mechanics."""

import random
from types import SimpleNamespace
import pytest
import numpy as np

from env.track import Track
from sac.track_pools import POOL_CONFIG, TRAIN_SEEDS, TEST_SEEDS
from sac.train_curriculum import episode_seed


def test_train_seeds_expanded():
    """Verify that TRAIN_SEEDS contains at least 25 distinct procedural seeds."""
    assert len(TRAIN_SEEDS) >= 25, f"Expected at least 25 training seeds, got {len(TRAIN_SEEDS)}"
    assert len(TRAIN_SEEDS) == 25, f"Expected exactly 25 training seeds, got {len(TRAIN_SEEDS)}"
    assert len(set(TRAIN_SEEDS)) == len(TRAIN_SEEDS), "TRAIN_SEEDS contains duplicates"
    assert set(range(1, 26)).issubset(set(TRAIN_SEEDS)), "Seeds 1..25 must be present in TRAIN_SEEDS"


def test_test_seeds_held_out_and_disjoint():
    """Verify that TEST_SEEDS remains unchanged (10 held-out seeds) and strictly disjoint from TRAIN_SEEDS."""
    assert TEST_SEEDS == list(range(1001, 1011)), "TEST_SEEDS must remain exactly seeds 1001..1010"
    assert len(TEST_SEEDS) == 10
    assert len(set(TEST_SEEDS)) == 10

    # Safety check: Zero overlap
    intersection = set(TRAIN_SEEDS).intersection(set(TEST_SEEDS))
    assert len(intersection) == 0, f"TRAIN and TEST pools must be disjoint, found overlap: {intersection}"
    assert set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS)


def test_episode_seed_sampling_all_25_tracks():
    """Verify that --train-pool 25 allows episode_seed to sample from all 25 training tracks."""
    args = SimpleNamespace(train_pool=25)
    tier = {"track_seed": 9999}

    # Slice must have exactly 25 seeds
    sampled_pool = TRAIN_SEEDS[:args.train_pool]
    assert len(sampled_pool) == 25

    # Seed random for determinism
    random.seed(42)

    # Sample enough times to guarantee coverage of all 25 seeds
    # Probability of missing a seed after 500 draws is 25 * (24/25)^500 ~ 25 * 1.3e-9 ~ 3.3e-8
    observed_seeds = set()
    for _ in range(500):
        s = episode_seed(args, tier)
        observed_seeds.add(s)
        assert s in sampled_pool, f"Sampled seed {s} not in TRAIN_SEEDS[:25]"
        assert s not in TEST_SEEDS, f"Sampled seed {s} belongs to held-out TEST_SEEDS!"

    # Verify every single seed 1..25 was sampled
    assert observed_seeds == set(range(1, 26))

    # Explicitly confirm the newly added seeds (21, 22, 23, 24, 25) were all sampled
    newly_added_seeds = {21, 22, 23, 24, 25}
    assert newly_added_seeds.issubset(observed_seeds), (
        f"Seeds {newly_added_seeds - observed_seeds} were not sampled from the expanded pool!"
    )


def test_episode_seed_train_pool_subsets():
    """Verify that episode_seed respects different train_pool sizes and default zero mode."""
    tier = {"track_seed": 777}

    # train_pool = 0 (fixed single-track mode)
    args_zero = SimpleNamespace(train_pool=0)
    for _ in range(20):
        assert episode_seed(args_zero, tier) == 777

    # train_pool = 5 (subset)
    args_5 = SimpleNamespace(train_pool=5)
    observed_5 = {episode_seed(args_5, tier) for _ in range(100)}
    assert observed_5 == set(range(1, 6))

    # train_pool = 10 (subset)
    args_10 = SimpleNamespace(train_pool=10)
    observed_10 = {episode_seed(args_10, tier) for _ in range(200)}
    assert observed_10 == set(range(1, 11))

    # train_pool = 20 (subset)
    args_20 = SimpleNamespace(train_pool=20)
    observed_20 = {episode_seed(args_20, tier) for _ in range(400)}
    assert observed_20 == set(range(1, 21))


def test_procedural_track_generation_all_25_seeds():
    """Verify that all 25 training seeds generate valid, drivable procedural tracks under POOL_CONFIG."""
    tk = POOL_CONFIG["track_kwargs"]
    for seed in TRAIN_SEEDS:
        track = Track(seed=seed, **tk)
        assert track.centerline is not None
        assert len(track.centerline) >= tk["n_ctrl"] * 20
        # Check start pose is on track
        start_x, start_y, heading = track.start_pose()
        assert track.is_on_track(start_x, start_y), f"Start pose of track seed {seed} is off-track"

# Docker 100-lap memory check

Image `rl-racing:p1` (built from this repo's `Dockerfile`, python:3.11-slim, CPU), single uvicorn worker.
Each "lap" = one `POST /simulation/rollout` (`max_steps=1000`, offline commentary), with a **distinct
`track_seed` per lap** so every call is a cache miss and does real compute. Run with
`ROLLOUT_RATE_LIMIT_PER_MIN=0` (otherwise the public 20/min limiter would return 429s). Memory is
`docker stats --no-stream` MemUsage, sampled every 10 laps.

| laps | before fix (MiB) | after fix (MiB) |
|---:|---:|---:|
| 0 | 180.9 | 180.6 |
| 10 | 212.5 | 210.2 |
| 20 | 235.2 | 232.8 |
| 30 | 258.0 | 255.8 |
| 40 | 280.1 | 277.5 |
| 50 | 301.0 | 298.4 |
| 60 | 322.9 | 321.8 |
| 70 | 332.7 | 332.0 |
| 80 | 334.2 | 332.7 |
| 90 | 334.0 | 333.0 |
| 100 | 333.6 | 332.2 |

Extended run (before fix), laps 120/140/160/180/200: 334.7 / 334.2 / 335.1 / 335.5 / 336.2 MiB.

## Verdict

- **Memory is flat after warm-up, not leaking.** It rises ~2.2 MiB/lap until the in-memory rollout LRU
  (`ROLLOUT_CACHE_MAX=64` entries) is full (~lap 64), then plateaus at ~333 MiB; laps 70-200 move by
  <3 MiB (noise). The ~150 MiB climb is the bounded cache, not growth.
- **One real leak found and fixed: the on-disk rollout cache was unbounded** (201 files / 110 MB in the
  container's writable layer after 200 laps). `inference/app.py` now prunes to `ROLLOUT_CACHE_DISK_MAX`
  files (default 256, oldest first). Verified: with `ROLLOUT_CACHE_DISK_MAX=50`, 100 laps left exactly
  50 files. The change is internal/additive; no endpoint shape or rollout result changed.
- To lower the plateau, set `ROLLOUT_CACHE_MAX` smaller (each cached rollout is ~2.2 MiB).

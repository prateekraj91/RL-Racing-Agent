# Cross-Track Failure Analysis

## Objective

Inspect zero-shot crashes on unseen hairpins and determine whether failures are primarily:
1. Under-steering
2. Late braking / excessive speed
3. Sliding

---

## 1. Test Setup

- Agent: V2
- Evaluation: zero-shot
- Track: unseen hairpin tracks
- Hairpin factors tested: 0.5, 0.35, 0.2
- Seeds tested: 20 per factor

### Results

| Hairpin factor | Completed laps | Timeouts | Crashes |
|---|---:|---:|---:|
| 0.5 | 0 | 6 | 14 |
| 0.35 | 0 | 8 | 12 |
| 0.2 | 0 | 8 | 12 |

---

## 2. Failure Case A: Seed 16

### Observations

- The upcoming corner becomes progressively tighter.
- Speed continues increasing as the corner approaches.
- Throttle remains close to +1.
- Signed distance from the centerline grows continuously.
- Steering eventually becomes strongly positive.
- Slip remains small.

### Diagnosis

This is primarily a late braking / excessive-speed failure.

The agent enters the tight corner too quickly. Once the trajectory is already displaced from the centerline, steering correction is insufficient to recover the trajectory.

Under-steering appears as a consequence of entering the corner too fast.

---

## 3. Failure Case B: Seed 10

### Observations

- Speed remains at approximately 4.0.
- Steering approaches +1.0.
- Signed distance continues increasing.
- Heading error remains large.
- Slip remains small.
- The vehicle eventually leaves the track.

### Diagnosis

This is a clear under-steering / trajectory-control failure.

The agent is applying almost maximum steering, but the vehicle continues moving away from the centerline.

The problem is therefore not simply insufficient steering action. The policy has already entered the corner at an inappropriate speed and trajectory.

---

## 4. Sliding Check

Slip remained relatively small in the representative crashes.

Therefore, sliding is not the primary failure mode observed in these cases.

---

## 5. Physics Sanity Check

A manual low-speed steering experiment showed that the vehicle can physically negotiate the hairpin.

Therefore, the hairpin is not inherently impossible for the vehicle model.

---

## 6. Final Conclusion

V2's zero-shot failures on unseen hairpins are primarily caused by inadequate anticipatory speed control.

When the agent approaches a tight corner at excessive speed, it cannot recover its trajectory even when steering becomes large. This produces the observed under-steering failure.

Sliding is not the primary cause, and the vehicle physics are capable of negotiating the hairpin.

### Main failure chain

corner approaches
→ speed remains too high
→ vehicle enters corner with poor trajectory
→ steering becomes insufficient to recover
→ cross-track error grows
→ crash
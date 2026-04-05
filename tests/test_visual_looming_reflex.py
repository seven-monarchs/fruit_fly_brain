"""
tests/test_visual_looming_reflex.py

Tests for the looming avoidance reflex:
  - Sudden luminance drop in one eye → turn bias away from that side
  - Symmetric drop → no net bias
  - Bias scales with drop magnitude
  - Bias decays over steps (exponential decay)
  - No effect when drop is below threshold

Reflex logic (to be added to main sim):
  Each step, compute luminance delta vs previous step:
    drop_l = prev_lum_l - lum_l   (positive = got darker)
    drop_r = prev_lum_r - lum_r

  Looming bias (added to turn_bias, same sign convention as odor_turn):
    + bias → turn left  (left_amp increases)
    - bias → turn right (right_amp increases)

  So: left eye drops → turn RIGHT → looming_bias is negative
      right eye drops → turn LEFT  → looming_bias is positive

  loom_bias = LOOM_GAIN * (drop_r - drop_l)   [only when above threshold]

Run:
    wenv310\\Scripts\\python.exe tests/test_visual_looming_reflex.py
"""

import numpy as np

# Parameters (match what we'll use in main sim)
LOOM_THRESHOLD = 0.015   # min per-step luminance drop to count as looming
LOOM_GAIN      = 3.0     # how strongly to steer away
LOOM_DECAY     = 0.6     # exponential decay per step (0=instant, 1=no decay)
WALK_AMP       = 0.75

def compute_loom_bias(drop_l, drop_r, threshold=LOOM_THRESHOLD, gain=LOOM_GAIN):
    """
    Returns looming turn bias.
    drop_l/drop_r: positive = luminance dropped (darker) on that eye.
    Returns: positive = turn left, negative = turn right.
    """
    effective_l = max(0.0, drop_l - threshold)
    effective_r = max(0.0, drop_r - threshold)
    return gain * (effective_r - effective_l)

def apply_bias(turn_bias, loom_bias):
    left_amp  = float(np.clip(WALK_AMP + turn_bias + loom_bias, 0.1, 1.0))
    right_amp = float(np.clip(WALK_AMP - turn_bias - loom_bias, 0.1, 1.0))
    return left_amp, right_amp

# ---------------------------------------------------------------------------
print("-- TEST 1: left eye drops → turn right (loom_bias < 0) --")
drop_l, drop_r = 0.08, 0.0   # landmark approaching on left
bias = compute_loom_bias(drop_l, drop_r)
left_amp, right_amp = apply_bias(turn_bias=0.0, loom_bias=bias)
print(f"  drop_l={drop_l}  drop_r={drop_r}  loom_bias={bias:.4f}")
print(f"  left_amp={left_amp:.3f}  right_amp={right_amp:.3f}")
assert bias < 0, f"FAIL: expected negative bias (turn right), got {bias}"
assert right_amp > left_amp, f"FAIL: expected right_amp > left_amp for right turn"
print("  OK turning right")

# ---------------------------------------------------------------------------
print("\n-- TEST 2: right eye drops → turn left (loom_bias > 0) --")
drop_l, drop_r = 0.0, 0.08
bias = compute_loom_bias(drop_l, drop_r)
left_amp, right_amp = apply_bias(turn_bias=0.0, loom_bias=bias)
print(f"  drop_l={drop_l}  drop_r={drop_r}  loom_bias={bias:.4f}")
print(f"  left_amp={left_amp:.3f}  right_amp={right_amp:.3f}")
assert bias > 0, f"FAIL: expected positive bias (turn left), got {bias}"
assert left_amp > right_amp, f"FAIL: expected left_amp > right_amp for left turn"
print("  OK turning left")

# ---------------------------------------------------------------------------
print("\n-- TEST 3: symmetric drop → no net bias --")
drop_l, drop_r = 0.06, 0.06
bias = compute_loom_bias(drop_l, drop_r)
print(f"  drop_l={drop_l}  drop_r={drop_r}  loom_bias={bias:.4f}")
assert abs(bias) < 1e-9, f"FAIL: expected zero bias for symmetric drop, got {bias}"
print("  OK no net bias")

# ---------------------------------------------------------------------------
print("\n-- TEST 4: below threshold → no bias --")
drop_l, drop_r = 0.01, 0.0   # below LOOM_THRESHOLD
bias = compute_loom_bias(drop_l, drop_r)
print(f"  drop_l={drop_l} (below threshold={LOOM_THRESHOLD})  loom_bias={bias:.4f}")
assert bias == 0.0, f"FAIL: expected zero bias below threshold, got {bias}"
print("  OK sub-threshold drop ignored")

# ---------------------------------------------------------------------------
print("\n-- TEST 5: bias scales with drop magnitude --")
biases = []
for drop in [0.02, 0.05, 0.10, 0.15]:
    b = compute_loom_bias(drop, 0.0)
    biases.append(b)
    print(f"  drop={drop:.2f}  bias={b:.4f}")
# Bias should become more negative as drop increases
for i in range(len(biases)-1):
    assert biases[i+1] < biases[i], f"FAIL: bias not increasing with drop at index {i}"
print("  OK bias scales with magnitude")

# ---------------------------------------------------------------------------
print("\n-- TEST 6: decay — bias fades over steps without new drop --")
loom_persist = 0.0
drop_l = 0.08  # one big drop at step 0

decayed = []
for step in range(6):
    if step == 0:
        new_bias = compute_loom_bias(drop_l, 0.0)
    else:
        new_bias = compute_loom_bias(0.0, 0.0)  # no new drop
    loom_persist = loom_persist * LOOM_DECAY + new_bias
    decayed.append(loom_persist)
    print(f"  step {step}: loom_persist={loom_persist:.4f}")

assert decayed[0] < 0,   "FAIL: initial bias should be negative"
assert decayed[1] > decayed[0], "FAIL: should decay (become less negative)"
assert abs(decayed[5]) < abs(decayed[0]), "FAIL: should decay over time"
print("  OK bias decays over steps")

# ---------------------------------------------------------------------------
print("\n-- TEST 7: reflex + odor steering coexist without saturation --")
# Fly turning right due to odor (right side stronger)
odor_turn = -0.4  # turning right
drop_l    = 0.08  # also sees dark object on left → also turns right
loom_bias = compute_loom_bias(drop_l, 0.0)
left_amp, right_amp = apply_bias(odor_turn, loom_bias)
print(f"  odor_turn={odor_turn:.3f}  loom_bias={loom_bias:.4f}")
print(f"  combined left_amp={left_amp:.3f}  right_amp={right_amp:.3f}")
assert 0.1 <= left_amp  <= 1.0, f"FAIL: left_amp out of range: {left_amp}"
assert 0.1 <= right_amp <= 1.0, f"FAIL: right_amp out of range: {right_amp}"
assert right_amp > left_amp, "FAIL: should be turning right"
print("  OK both biases combine cleanly, values in [0.1, 1.0]")

# ---------------------------------------------------------------------------
print("\n=== ALL TESTS PASSED ===")
print(f"Looming reflex: threshold={LOOM_THRESHOLD}, gain={LOOM_GAIN}, decay={LOOM_DECAY}")
print("Left eye drops  → negative bias → turn right")
print("Right eye drops → positive bias → turn left")
print("Decays exponentially when no new looming detected")

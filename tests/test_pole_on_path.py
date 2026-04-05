"""
tests/test_pole_on_path.py

Test: single pole placed DIRECTLY on the fly's natural path.

Fly starts at (0,0), heading 0.588 rad (~34°), food at (18,12).
Natural path direction vector: (18,12)/||...|| ≈ (0.832, 0.555)
Pole at ~8mm along that path: (6.7, 4.4) → we use (7.0, 4.5)

Hypothesis:
  - Front-on pole → both eyes equally darkened → reflex bias ≈ 0
  - Fly should drift slightly (odor pulls it toward food which is off-center)
  - Without reflex: fly may approach pole and stall (keeps straight into it)
  - With reflex: symmetric signal → minimal turn; BUT odor asymmetry may help

Three conditions:
  A) No pole (baseline)
  B) Pole on path + NO reflex
  C) Pole on path + WITH reflex

Run:
    wenv310\\Scripts\\python.exe tests/test_pole_on_path.py
"""

import numpy as np
import flygym.preprogrammed as preprogrammed
from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

FOOD_POS       = np.array([18.0, 12.0, 0.0])
PHYSICS_TIMESTEP = 1e-4
WALK_AMP       = 0.75
ODOR_TURN_K    = 2.5
FEED_DIST      = 1.2
PHYSICS_STEPS_PER_DECISION = 250

LOOM_THRESHOLD = 0.005
LOOM_GAIN      = 6.0
LOOM_DECAY     = 0.6
LOOM_BG_ALPHA  = 0.85

# Single cylinder placed DIRECTLY on the fly's natural path
# Path direction from (0,0) at heading 0.588 rad: (cos,sin) ≈ (0.832, 0.555)
# At ~8mm along path: (6.7, 4.4) → rounded to (7.0, 4.5)
POLE_ON_PATH = [
    {'type':'cylinder','pos':[7.0, 4.5, 5.0],'size':[1.0, 4.0],
     'rgba':[0.1, 0.07, 0.04, 1],'contype':0,'conaffinity':0},
]

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]

N_STEPS = 250


def run_scenario(geoms, with_reflex, label):
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2,
        marker_colors=[],
    )
    for g in geoms:
        arena.root_element.worldbody.add('geom', **g)

    fly = Fly(
        spawn_pos=(0, 0, 0.2), spawn_orientation=(0, 0, 0.588),
        contact_sensor_placements=csp,
        enable_olfaction=True, enable_adhesion=True,
        draw_adhesion=False, enable_vision=True,
    )
    sim = HybridTurningController(fly=fly, arena=arena, timestep=PHYSICS_TIMESTEP)
    obs, _ = sim.reset()

    _names = list(sim.physics.named.data.qpos.axes[0].names)
    root_joint = next((n for n in _names if n.endswith('/') and 'dust' not in n), None)

    loom_bg      = None
    loom_persist = 0.0

    dist_history = []
    loom_history = []
    x_history    = []
    y_history    = []
    sig_l_history = []
    sig_r_history = []
    reached_food = False

    for step in range(N_STEPS):
        odor       = obs['odor_intensity'][0]
        left_odor  = float(odor[0] + odor[2]) / 2.0
        right_odor = float(odor[1] + odor[3]) / 2.0
        total_odor = left_odor + right_odor
        lr_asym    = (right_odor - left_odor) / (total_odor + 1e-9)
        odor_turn  = float(np.tanh(lr_asym * 20.0) * ODOR_TURN_K)

        sig_l = sig_r = 0.0
        if with_reflex:
            vis = obs.get('vision', None)
            if vis is not None:
                curr = vis.mean(axis=2)
                if loom_bg is None:
                    loom_bg = curr.copy()
                delta   = loom_bg - curr
                delta_t = np.where(delta > LOOM_THRESHOLD, delta - LOOM_THRESHOLD, 0.0)
                sig_l   = float(delta_t[0].mean())
                sig_r   = float(delta_t[1].mean())
                loom_new     = -LOOM_GAIN * (sig_l - sig_r)
                loom_persist = loom_persist * LOOM_DECAY + loom_new
                loom_bg      = loom_bg * LOOM_BG_ALPHA + curr * (1.0 - LOOM_BG_ALPHA)

        turn_bias = odor_turn + loom_persist
        ctrl = np.array([
            float(np.clip(WALK_AMP + turn_bias, 0.1, 1.0)),
            float(np.clip(WALK_AMP - turn_bias, 0.1, 1.0)),
        ])

        for _ in range(PHYSICS_STEPS_PER_DECISION):
            obs, _, _, _, _ = sim.step(ctrl)

        qpos = sim.physics.named.data.qpos[root_joint]
        pos  = np.array([float(qpos[0]), float(qpos[1])])
        dist = float(np.linalg.norm(pos - FOOD_POS[:2]))

        dist_history.append(dist)
        loom_history.append(loom_persist)
        x_history.append(float(pos[0]))
        y_history.append(float(pos[1]))
        sig_l_history.append(sig_l)
        sig_r_history.append(sig_r)

        if dist < FEED_DIST:
            reached_food = True
            print(f"  [{label}] reached food at step {step+1}!")
            break

    final_dist = dist_history[-1]
    min_dist   = min(dist_history)
    max_loom   = max(abs(l) for l in loom_history) if loom_history else 0.0
    print(f"  [{label}] steps={len(dist_history)}  min={min_dist:.2f}mm  "
          f"final={final_dist:.2f}mm  reached={reached_food}  "
          f"max_loom={max_loom:.3f}")

    return dist_history, min_dist, reached_food, loom_history, x_history, y_history, sig_l_history, sig_r_history


# ---------------------------------------------------------------------------
print("=" * 65)
print("Test: pole placed DIRECTLY on fly's path")
print("Fly: (0,0) heading=0.588 rad  Food: (18,12)")
print("Pole: (7.0, 4.5) — on path at ~8mm  Size: r=1mm, half-h=4mm")
print("=" * 65)

print("\n-- Condition A: no pole (baseline) --")
r_a = run_scenario([], with_reflex=False, label="A: no pole")

print("\n-- Condition B: pole on path + NO reflex --")
r_b = run_scenario(POLE_ON_PATH, with_reflex=False, label="B: pole, no reflex")

print("\n-- Condition C: pole on path + WITH reflex --")
r_c = run_scenario(POLE_ON_PATH, with_reflex=True,  label="C: pole + reflex")

hist_a, min_a, reached_a, loom_a, xa, ya, sla, sra = r_a
hist_b, min_b, reached_b, loom_b, xb, yb, slb, srb = r_b
hist_c, min_c, reached_c, loom_c, xc, yc, slc, src = r_c

# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)
print(f"  A (no pole):       min={min_a:.2f}mm  reached={reached_a}")
print(f"  B (pole,no reflex):min={min_b:.2f}mm  reached={reached_b}")
print(f"  C (pole + reflex): min={min_c:.2f}mm  reached={reached_c}")

if reached_c and not reached_b:
    print("\n  STRONG WIN: reflex allowed fly to navigate around pole!")
elif min_c < min_b:
    print(f"\n  IMPROVEMENT: reflex got {min_b-min_c:.2f}mm closer to food")
elif min_c == min_b and not reached_c:
    print("\n  STUCK: both conditions got stuck — front-on signal is symmetric")
    print("  (expected: both eyes equally darkened → zero reflex asymmetry)")
else:
    print(f"\n  MIXED: C={min_c:.2f}mm vs B={min_b:.2f}mm")

# Symmetry analysis — did reflex fire asymmetrically?
max_loom_c = max(abs(l) for l in loom_c) if loom_c else 0.0
print(f"\n  Max loom_persist (C): {max_loom_c:.4f}")

# Check if sig_l ≈ sig_r (front-on symmetry)
nonzero_steps = [(i, slc[i], src[i]) for i in range(len(slc))
                 if slc[i] > 0.001 or src[i] > 0.001]
if nonzero_steps:
    avg_l = np.mean([s[1] for s in nonzero_steps])
    avg_r = np.mean([s[2] for s in nonzero_steps])
    asym  = (avg_l - avg_r) / (avg_l + avg_r + 1e-9)
    print(f"  Active steps: {len(nonzero_steps)}")
    print(f"  Mean sig_l={avg_l:.5f}  sig_r={avg_r:.5f}  asymmetry={asym:+.4f}")
    if abs(asym) < 0.1:
        print("  -> SYMMETRIC (front-on): reflex gives no useful turn direction")
    else:
        side = "LEFT eye dominates (turn RIGHT)" if asym > 0 else "RIGHT eye dominates (turn LEFT)"
        print(f"  -> ASYMMETRIC: {side}")
else:
    print("  Reflex never fired above threshold")

# Path comparison — did trajectories diverge?
print(f"\n-- Path comparison (first 30 steps) --")
print(f"  {'step':>4}  {'x_B':>6} {'y_B':>6}  {'x_C':>6} {'y_C':>6}  {'dy':>6}  loom_C")
for i in range(min(30, len(xb), len(xc))):
    dy = yc[i] - yb[i]
    loom = loom_c[i] if i < len(loom_c) else 0.0
    marker = ' <--' if abs(loom) > 0.05 else ''
    print(f"  {i+1:>4}  {xb[i]:>6.2f} {yb[i]:>6.2f}  {xc[i]:>6.2f} {yc[i]:>6.2f}  "
          f"{dy:>+6.2f}  {loom:>+.3f}{marker}")

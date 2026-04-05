"""
tests/test_wall_before_food.py

Test: a wall placed between the fly and the food.

Food at (18, 12). Wall at x=14, spanning y=6..18 (12mm wide, 8mm tall).
Wall is a ghost geom (contype=0) — no physics collision, purely visual.

Fly heading 0.588 rad (~34 deg) from (0,0) toward food.

Three conditions:
  A) No wall         -> baseline
  B) Wall + no reflex -> fly walks straight through (odor unaffected)
  C) Wall + reflex    -> symmetric wall darkens both eyes equally
                         -> reflex asymmetry near zero
                         -> fly still walks through

Expected:
  - All three reach food in similar steps
  - sig_l ≈ sig_r when approaching head-on (symmetric signal)
  - No meaningful directional bias from reflex
  - Fly does NOT go around — it walks through (ghost wall + no asymmetric signal)

Run:
    wenv310\\Scripts\\python.exe tests/test_wall_before_food.py
"""

import numpy as np
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

# Wall: thin box, x=14mm, spanning y=6..18, z=0..8
# size=[half_x, half_y, half_z] -> 0.3mm thick, 6mm half-span, 4mm half-height
# pos z=4 -> base at z=0 (floor level)
WALL = [
    {'type':'box', 'pos':[14.0, 12.0, 6.0], 'size':[0.3, 6.0, 4.0],
     'rgba':[0.1, 0.07, 0.04, 1.0], 'contype':0, 'conaffinity':0},
]

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]

N_STEPS = 300


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

    dist_history  = []
    loom_history  = []
    x_history     = []
    y_history     = []
    sig_l_history = []
    sig_r_history = []
    reached_food  = False

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
          f"final={final_dist:.2f}mm  reached={reached_food}  max_loom={max_loom:.3f}")

    return dist_history, min_dist, reached_food, loom_history, x_history, y_history, sig_l_history, sig_r_history


# ---------------------------------------------------------------------------
print("=" * 65)
print("Test: wall placed in front of food (ghost, no collision)")
print("Fly:  (0,0) heading=0.588 rad    Food: (18,12)")
print("Wall: x=14mm, y=6..18, z=2..10  (12mm wide, 8mm tall, 0.6mm thick, base at z=2)")
print("=" * 65)

print("\n-- Condition A: no wall (baseline) --")
r_a = run_scenario([], with_reflex=False, label="A: no wall")

print("\n-- Condition B: wall + NO reflex --")
r_b = run_scenario(WALL, with_reflex=False, label="B: wall, no reflex")

print("\n-- Condition C: wall + WITH reflex --")
r_c = run_scenario(WALL, with_reflex=True,  label="C: wall + reflex")

hist_a, min_a, reached_a, loom_a, xa, ya, sla, sra = r_a
hist_b, min_b, reached_b, loom_b, xb, yb, slb, srb = r_b
hist_c, min_c, reached_c, loom_c, xc, yc, slc, src = r_c

# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)
print(f"  A (no wall):       min={min_a:.2f}mm  reached={reached_a}  steps={len(hist_a)}")
print(f"  B (wall,no reflex):min={min_b:.2f}mm  reached={reached_b}  steps={len(hist_b)}")
print(f"  C (wall + reflex): min={min_c:.2f}mm  reached={reached_c}  steps={len(hist_c)}")

if not reached_b and not reached_c:
    print("\n  STUCK: fly blocked by wall in both conditions (no collision bypass)")
elif reached_b and reached_c:
    diff = len(hist_c) - len(hist_b)
    if diff > 3:
        print(f"\n  SLOWER with reflex (+{diff} steps) — symmetric signal adds noise")
    elif diff < -3:
        print(f"\n  FASTER with reflex ({diff} steps) — reflex helped navigate")
    else:
        print(f"\n  SAME: wall has no effect (fly walks through ghost wall, step diff={diff:+d})")

# Symmetry analysis near wall (steps when fly x > 10mm)
wall_steps = [(i, slc[i], src[i], loom_c[i])
              for i in range(len(slc)) if xc[i] > 10.0]
print(f"\n  Steps with fly x > 10mm (approaching wall zone): {len(wall_steps)}")
if wall_steps:
    avg_l   = np.mean([s[1] for s in wall_steps])
    avg_r   = np.mean([s[2] for s in wall_steps])
    avg_loom = np.mean([abs(s[3]) for s in wall_steps])
    total   = avg_l + avg_r
    asym    = (avg_l - avg_r) / (total + 1e-9)
    print(f"  Mean sig_l={avg_l:.5f}  sig_r={avg_r:.5f}")
    print(f"  Asymmetry={asym:+.4f}  mean|loom_persist|={avg_loom:.4f}")
    if abs(asym) < 0.05:
        print("  -> Nearly SYMMETRIC: wall darkens both eyes equally")
        print("     Reflex gives no useful steering direction")
        print("     Fly relies entirely on odor to navigate (walks through ghost wall)")
    else:
        side = "left" if asym > 0 else "right"
        print(f"  -> ASYMMETRIC ({side} eye stronger): reflex pushes fly to one side")

# Step-by-step near the wall
print(f"\n-- Step detail when x_C > 10mm (approaching/crossing wall at x=14) --")
print(f"  {'step':>4}  {'x_C':>6} {'y_C':>6}  {'sig_l':>8} {'sig_r':>8}  {'loom':>7}  {'asym':>7}")
for i, sl, sr, loom in wall_steps[:25]:
    total = sl + sr
    asym  = (sl - sr) / (total + 1e-9)
    print(f"  {i+1:>4}  {xc[i]:>6.2f} {yc[i]:>6.2f}  {sl:>8.5f} {sr:>8.5f}  "
          f"{loom:>+7.4f}  {asym:>+7.4f}")

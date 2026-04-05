"""
tests/test_looming_integration.py

Full closed-loop integration test for the looming avoidance reflex.

Runs the fly through 200 steps of odor-guided navigation WITH a blocking
cylinder on the path, comparing three conditions:

  A) No cylinder          -> baseline, fly reaches food
  B) Cylinder + no reflex -> fly may get stuck
  C) Cylinder + reflex    -> fly should get closer to food than (B)

No Brian2 (too slow) — uses odor steering only + loom reflex.
This isolates whether the reflex itself helps navigation.

Steering logic mirrors fly_brain_body_simulation.py exactly:
  turn_bias = odor_turn [+ loom_persist]
  ctrl = [clip(WALK_AMP + turn_bias), clip(WALK_AMP - turn_bias)]

Run:
    wenv310\\Scripts\\python.exe tests/test_looming_integration.py
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
FEED_DIST            = 1.2
PHYSICS_STEPS_PER_DECISION = 250   # microsteps per decision (same as main sim)

# Looming reflex params (same as main sim)
LOOM_THRESHOLD = 0.005
LOOM_GAIN      = 6.0
LOOM_DECAY     = 0.6
LOOM_BG_ALPHA  = 0.85

# Cylinders 2mm LEFT of fly's natural path — ghost objects, no collision
# Fly passes ~2mm to their right → right eye darkens → reflex steers left (detour)
CYLS = [
    {'type':'cylinder','pos':[5.0,5.5,5.0],'size':[1.0,4.0],
     'rgba':[0.1,0.07,0.04,1],'contype':0,'conaffinity':0},
    {'type':'cylinder','pos':[11.0,9.5,5.0],'size':[1.0,4.0],
     'rgba':[0.1,0.07,0.04,1],'contype':0,'conaffinity':0},
]

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]

N_STEPS = 200

def run_scenario(with_cylinder, with_reflex, label):
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2,
        marker_colors=[],
    )
    if with_cylinder:
        for c in CYLS:
            arena.root_element.worldbody.add('geom', **c)

    fly = Fly(
        spawn_pos=(0, 0, 0.2), spawn_orientation=(0, 0, 0.588),
        contact_sensor_placements=csp,
        enable_olfaction=True, enable_adhesion=True,
        draw_adhesion=False, enable_vision=True,
    )
    sim = HybridTurningController(fly=fly, arena=arena, timestep=PHYSICS_TIMESTEP)
    obs, _ = sim.reset()

    # World position via freejoint qpos
    _names = list(sim.physics.named.data.qpos.axes[0].names)
    root_joint = next((n for n in _names if n.endswith('/') and 'dust' not in n), None)

    # Looming state
    loom_bg      = None
    loom_persist = 0.0

    dist_history = []
    loom_history = []
    min_dist     = 999.0
    reached_food = False

    for step in range(N_STEPS):
        # Odor steering
        odor      = obs['odor_intensity'][0]
        left_odor  = float(odor[0] + odor[2]) / 2.0
        right_odor = float(odor[1] + odor[3]) / 2.0
        total_odor = left_odor + right_odor
        lr_asym    = (right_odor - left_odor) / (total_odor + 1e-9)
        odor_turn  = float(np.tanh(lr_asym * 20.0) * ODOR_TURN_K)

        # Looming reflex
        if with_reflex:
            vis = obs.get('vision', None)
            if vis is not None:
                curr = vis.mean(axis=2)   # (2, 721)
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
        if dist < min_dist:
            min_dist = dist
        if dist < FEED_DIST:
            reached_food = True
            print(f"  [{label}] reached food at step {step+1}!")
            break

    final_dist = dist_history[-1]
    min_dist   = min(dist_history)
    print(f"  [{label}] steps={len(dist_history)}  start={dist_history[0]:.2f}mm"
          f"  min={min_dist:.2f}mm  final={final_dist:.2f}mm"
          f"  reached={reached_food}"
          f"  max_loom={max(abs(l) for l in loom_history):.3f}")
    return dist_history, min_dist, reached_food, loom_history


# ---------------------------------------------------------------------------
print("=" * 60)
print("Integration test: looming reflex vs cylinder obstacle")
print("=" * 60)

print("\n-- Condition A: no cylinder (baseline) --")
hist_a, min_a, reached_a, loom_a = run_scenario(
    with_cylinder=False, with_reflex=False, label="A: no cyl")

print("\n-- Condition B: cylinder + NO reflex --")
hist_b, min_b, reached_b, loom_b = run_scenario(
    with_cylinder=True, with_reflex=False, label="B: cyl no reflex")

print("\n-- Condition C: cylinder + WITH reflex --")
hist_c, min_c, reached_c, loom_c = run_scenario(
    with_cylinder=True, with_reflex=True, label="C: cyl + reflex")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  A (no cylinder):       min_dist={min_a:.2f}mm  reached={reached_a}")
print(f"  B (cyl, no reflex):    min_dist={min_b:.2f}mm  reached={reached_b}")
print(f"  C (cyl + reflex):      min_dist={min_c:.2f}mm  reached={reached_c}")
print()

# Did reflex help at all?
if reached_c and not reached_b:
    print("  STRONG WIN: reflex allowed fly to reach food, no-reflex got stuck")
elif min_c < min_b:
    improvement = min_b - min_c
    print(f"  IMPROVEMENT: reflex got {improvement:.2f}mm closer to food than no-reflex")
elif min_c == min_b:
    print("  NEUTRAL: reflex made no difference (fly may not have passed close to cylinder)")
else:
    print(f"  WORSE: reflex actually hurt ({min_c:.2f}mm vs {min_b:.2f}mm) — gain may be too strong")

# Check reflex actually fired
max_loom = max(abs(l) for l in loom_c)
print(f"  Max loom_persist in C: {max_loom:.4f}")
if max_loom > 0.01:
    print("  Reflex fired during run")
else:
    print("  WARNING: reflex never fired — cylinder may not have been in visual field")

# ---------------------------------------------------------------------------
print("\n-- Step-by-step: first 60 steps near cylinder zone --")
print(f"  {'step':>4}  {'dist_B':>8}  {'dist_C':>8}  {'loom_C':>9}  diff")
for i in range(min(60, len(hist_b), len(hist_c))):
    diff = hist_b[i] - hist_c[i]
    marker = ' <-- loom active' if abs(loom_c[i]) > 0.05 else ''
    print(f"  {i+1:>4}  {hist_b[i]:>8.2f}  {hist_c[i]:>8.2f}  {loom_c[i]:>9.4f}  {diff:>+.2f}{marker}")

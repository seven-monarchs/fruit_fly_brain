"""
tests/test_solid_wall_navigation.py

Solid wall (Fix 2: soft solimp/solref) — does the fly navigate around it?

Setup:
  - Fly: (0,0) heading=0.588 rad   Food: (18,12)
  - Wall: x=14mm, spans y=4..20 (16mm wide), base at z=1
    -> Covers the fly's full approach path (y~9-10 at x=14)
    -> Gap only below y=4 or above y=20
    -> Fly MUST make a significant turn to get around

Conditions:
  A) No wall             -> baseline steps
  B) Solid wall, no reflex -> fly bounces, odor-only steering
  C) Solid wall + reflex   -> does asymmetric wall signal help steer to gap?

Run:
    wenv310\\Scripts\\python.exe tests/test_solid_wall_navigation.py
"""

import numpy as np
from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

FOOD_POS         = np.array([18.0, 12.0, 0.0])
PHYSICS_TIMESTEP = 1e-4
WALK_AMP         = 0.75
ODOR_TURN_K      = 2.5
FEED_DIST        = 1.2
PHYSICS_STEPS    = 250
N_STEPS          = 350

LOOM_THRESHOLD = 0.005
LOOM_GAIN      = 6.0
LOOM_DECAY     = 0.6
LOOM_BG_ALPHA  = 0.85

# Solid wall, y=4..20 (half-span=8, centered at y=12), base at z=1
# Wide and low enough that the fly's natural curved path hits it
# Gap only below y=4 — fly must turn hard right to get around
WALL = {
    'type': 'box',
    'pos':  [14.0, 12.0, 5.0],
    'size': [0.3, 8.0, 4.0],
    'rgba': [0.15, 0.10, 0.05, 1.0],
    'contype': 1, 'conaffinity': 1,
    'solimp': '0.9 0.999 0.001 0.5 2',
    'solref': '0.02 1',
}

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]


def run_scenario(use_wall, with_reflex, label):
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2,
        marker_colors=[],
    )
    if use_wall:
        arena.root_element.worldbody.add('geom', **WALL)

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
    sig_l = sig_r = 0.0

    x_hist     = []
    y_hist     = []
    loom_hist  = []
    dist_hist  = []
    sig_l_hist = []
    sig_r_hist = []
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

        try:
            for _ in range(PHYSICS_STEPS):
                obs, _, _, _, _ = sim.step(ctrl)
        except Exception as e:
            print(f"  [{label}] PHYSICS CRASH at step {step+1}: {e}")
            break

        qpos = sim.physics.named.data.qpos[root_joint]
        pos  = np.array([float(qpos[0]), float(qpos[1])])
        dist = float(np.linalg.norm(pos - FOOD_POS[:2]))

        x_hist.append(float(pos[0]))
        y_hist.append(float(pos[1]))
        loom_hist.append(loom_persist)
        dist_hist.append(dist)
        sig_l_hist.append(sig_l)
        sig_r_hist.append(sig_r)

        if dist < FEED_DIST:
            reached_food = True
            break

    min_dist   = min(dist_hist) if dist_hist else 999
    final_dist = dist_hist[-1] if dist_hist else 999
    max_loom   = max(abs(v) for v in loom_hist) if loom_hist else 0.0

    status = f"reached food at step {len(dist_hist)}" if reached_food \
             else f"did NOT reach food  final={final_dist:.1f}mm  min={min_dist:.1f}mm"
    print(f"  [{label}]  {status}  max_loom={max_loom:.3f}")

    return x_hist, y_hist, loom_hist, dist_hist, sig_l_hist, sig_r_hist, reached_food


# ---------------------------------------------------------------------------
print("=" * 65)
print("Solid wall navigation  (soft solimp/solref, adhesion on)")
print("Wall: x=14mm, y=4..20  gap only below y=4  base at z=1")
print("Food: (18,12)   Fly: (0,0) heading=0.588 rad")
print("=" * 65)

print("\n-- A: no wall (baseline) --")
xa, ya, la, da, sla, sra, ra = run_scenario(False, False, "A: no wall")

print("\n-- B: solid wall, no reflex --")
xb, yb, lb, db, slb, srb, rb = run_scenario(True, False, "B: solid wall, no reflex")

print("\n-- C: solid wall + looming reflex --")
xc, yc, lc, dc, slc, src, rc = run_scenario(True, True, "C: solid wall + reflex")

# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("ANALYSIS")
print("=" * 65)

steps_a = len(da)
steps_b = len(db)
steps_c = len(dc)

print(f"\n  Steps to finish: A={steps_a}  B={steps_b}  C={steps_c}")

if rb and rc:
    diff = steps_c - steps_b
    if diff > 5:
        print(f"  Reflex slower (+{diff} steps) — pushed fly further from gap")
    elif diff < -5:
        print(f"  Reflex faster ({diff} steps) — helped find gap!")
    else:
        print(f"  Similar speed (diff={diff:+d})")
elif rb and not rc:
    print("  Reflex PREVENTED food reach — pushed fly wrong way")
elif not rb and rc:
    print("  Reflex ENABLED food reach — found gap that odor alone missed!")
elif not rb and not rc:
    print("  Both stuck — wall fully blocks. Gap too narrow or odor too strong toward wall.")

# Gap zone analysis (y < 4 = below wall)
gap_y = 4.0
b_gap_steps = sum(1 for y in yb if y < gap_y)
c_gap_steps = sum(1 for y in yc if y < gap_y)
print(f"\n  Steps in gap zone (y < {gap_y}mm): B={b_gap_steps}  C={c_gap_steps}")
if c_gap_steps > b_gap_steps:
    print("  -> Reflex successfully pushed fly toward the gap!")
elif c_gap_steps < b_gap_steps:
    print("  -> Reflex pushed fly away from gap (wrong direction)")
else:
    print("  -> Same gap usage (reflex had no directional effect here)")

# Where did fly B stall vs fly C diverge?
print(f"\n-- Per-step path near wall (x > 11mm) --")
print(f"  {'step':>4}  {'x_B':>6} {'y_B':>6}  {'x_C':>6} {'y_C':>6}  "
      f"{'loom_C':>7}  {'sigL':>6} {'sigR':>6}  note")

n = max(len(xb), len(xc))
xb_p = xb + [None]*(n-len(xb))
yb_p = yb + [None]*(n-len(yb))
xc_p = xc + [None]*(n-len(xc))
yc_p = yc + [None]*(n-len(yc))
lc_p = lc + [None]*(n-len(lc))
sl_p = slc + [None]*(n-len(slc))
sr_p = src + [None]*(n-len(src))

printed = 0
for i in range(n):
    in_zone = (xb_p[i] is not None and xb_p[i] > 11.0) or \
              (xc_p[i] is not None and xc_p[i] > 11.0)
    if not in_zone:
        continue

    fmt = lambda v, w=6, d=2: f"{v:{w}.{d}f}" if v is not None else ' '*w
    note = []
    if yc_p[i] is not None and yc_p[i] < gap_y:
        note.append("IN GAP")
    if lc_p[i] is not None and abs(lc_p[i]) > 0.3:
        note.append(f"LOOM {'R' if lc_p[i] < 0 else 'L'}")
    if xb_p[i] is not None and i+1 < len(xb_p) and xb_p[i+1] is not None:
        if abs(xb_p[i+1] - xb_p[i]) < 0.05:
            note.append("B-STALL")

    print(f"  {i+1:>4}  {fmt(xb_p[i])} {fmt(yb_p[i])}  "
          f"{fmt(xc_p[i])} {fmt(yc_p[i])}  "
          f"{fmt(lc_p[i], 7, 3)}  "
          f"{fmt(sl_p[i], 6, 4)} {fmt(sr_p[i], 6, 4)}  "
          f"{' '.join(note)}")
    printed += 1
    if printed >= 60:
        print(f"  ... (truncated, {n - i - 1} more steps)")
        break

"""
tests/test_wall_base_sweep.py

Sweep wall base z height to find the minimum that:
  1. Blocks the fly (doesn't let it walk under)
  2. Doesn't crash with BADQACC

Tests base z values: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
Uses odor-only navigation (no flyvis) for speed — just checking physics stability.

Run:
    wenv310\\Scripts\\python.exe tests/test_wall_base_sweep.py
"""

from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

import numpy as np

FOOD_POS         = np.array([18.0, 12.0, 0.0])
SPAWN_POS        = (0, 0, 0.2)
SPAWN_HEADING    = (0, 0, -1.2)
PHYSICS_TIMESTEP = 1e-4
WALK_AMP         = 0.75
ODOR_TURN_K      = 2.5
FEED_DIST        = 1.2
PHYSICS_STEPS    = 250
N_STEPS          = 150   # only need to reach x=14

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]

BASE_HEIGHTS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

print("=" * 65)
print("Wall base height sweep — odor only, heading=-1.2 rad")
print(f"{'Base z':>8}  {'Outcome':>30}  {'y at x=14':>10}  {'Under?':>7}")
print("=" * 65)

for base_z in BASE_HEIGHTS:
    pos_z = 4.0 + base_z   # pos_z = half_height + base_z
    wall = {
        'type': 'box', 'pos': [14.0, 12.0, pos_z], 'size': [0.3, 8.0, 4.0],
        'rgba': [0.15, 0.10, 0.05, 1.0], 'contype': 1, 'conaffinity': 1,
        'solimp': '0.9 0.999 0.001 0.5 2', 'solref': '0.02 1',
    }

    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2, marker_colors=[],
    )
    arena.root_element.worldbody.add('geom', **wall)

    fly = Fly(spawn_pos=SPAWN_POS, spawn_orientation=SPAWN_HEADING,
              contact_sensor_placements=csp, enable_olfaction=True,
              enable_adhesion=True, draw_adhesion=False, enable_vision=False)
    sim = HybridTurningController(fly=fly, arena=arena, timestep=PHYSICS_TIMESTEP)
    obs, _ = sim.reset()
    _names = list(sim.physics.named.data.qpos.axes[0].names)
    root_joint = next((n for n in _names if n.endswith('/') and 'dust' not in n), None)

    x_hist = []; y_hist = []
    crashed = False
    reached = False

    for step in range(N_STEPS):
        odor      = obs['odor_intensity'][0]
        lr_asym   = (float(odor[1]+odor[3]) - float(odor[0]+odor[2])) / (sum(odor)+1e-9)
        odor_turn = float(np.tanh(lr_asym * 20.0) * ODOR_TURN_K)
        ctrl = np.array([np.clip(WALK_AMP+odor_turn, 0.1, 1.0),
                         np.clip(WALK_AMP-odor_turn, 0.1, 1.0)])
        try:
            for _ in range(PHYSICS_STEPS):
                obs, _, _, _, _ = sim.step(ctrl)
        except Exception:
            crashed = True
            break

        qpos = sim.physics.named.data.qpos[root_joint]
        pos  = np.array([float(qpos[0]), float(qpos[1])])
        x_hist.append(pos[0]); y_hist.append(pos[1])
        if np.linalg.norm(pos - FOOD_POS[:2]) < FEED_DIST:
            reached = True
            break

    x = np.array(x_hist); y = np.array(y_hist)
    crossing = np.where((x[:-1] < 14) & (x[1:] >= 14))[0]

    if crashed:
        outcome = "CRASH (BADQACC)"
        y_cross = "-"
        under   = "-"
    elif len(crossing):
        ci      = crossing[0]
        y_c     = y[ci]
        y_cross = f"{y_c:.2f}mm"
        under   = "YES (walked under)" if y_c > 4.0 else "NO  (used gap)"
        outcome = f"Crossed x=14 at step {ci}"
    else:
        outcome = "Blocked (never crossed x=14)"
        y_cross = "-"
        under   = "NO"

    print(f"  {base_z:.2f}mm  {outcome:>30}  {y_cross:>10}  {under}")

print("=" * 65)
print("Target: lowest base_z that says 'Blocked' without CRASH")

"""
tests/test_solid_wall_stability.py

Find a configuration that allows a SOLID wall without BADQACC.

Root cause of BADQACC with solid geoms + fly:
  - The fly uses adhesion actuators on tarsi (sticky feet)
  - When a tarsus contacts a vertical surface, adhesion force acts horizontally
  - This creates torques the solver can't resolve in one 0.1ms timestep -> NaN

Three fixes to test, in order of cost:
  1. enable_adhesion=False       -- remove adhesion entirely (cheap, lossless for navigation)
  2. solimp/solref softening     -- slow contact force buildup
  3. timestep = 5e-5             -- 2x more microsteps per decision

Tests a solid box wall at x=14 with each fix.
If the fly reaches food -> fix works.
If PhysicsError -> fix not sufficient.

Run:
    wenv310\\Scripts\\python.exe tests/test_solid_wall_stability.py
"""

import numpy as np
from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

FOOD_POS         = np.array([18.0, 12.0, 0.0])
WALK_AMP         = 0.75
ODOR_TURN_K      = 2.5
FEED_DIST        = 1.2
N_STEPS          = 250

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]


def make_wall(soft=False):
    w = {'type': 'box', 'pos': [14.0, 12.0, 6.0], 'size': [0.3, 6.0, 4.0],
         'rgba': [0.15, 0.1, 0.05, 1.0], 'contype': 1, 'conaffinity': 1}
    if soft:
        w['solimp'] = '0.9 0.999 0.001 0.5 2'
        w['solref'] = '0.02 1'
    return w


def run_trial(label, timestep=1e-4, adhesion=True, soft_wall=False, microsteps=250):
    print(f"\n  [{label}]  dt={timestep}  adhesion={adhesion}  soft_wall={soft_wall}  microsteps={microsteps}")
    try:
        arena = OdorArena(
            odor_source=FOOD_POS[np.newaxis],
            peak_odor_intensity=np.array([[500.0, 0.0]]),
            diffuse_func=lambda x: x**-2,
            marker_colors=[],
        )
        arena.root_element.worldbody.add('geom', **make_wall(soft=soft_wall))

        fly = Fly(
            spawn_pos=(0, 0, 0.2), spawn_orientation=(0, 0, 0.588),
            contact_sensor_placements=csp,
            enable_olfaction=True,
            enable_adhesion=adhesion,
            draw_adhesion=False,
            enable_vision=False,   # not needed for stability test
        )
        sim = HybridTurningController(fly=fly, arena=arena, timestep=timestep)
        obs, _ = sim.reset()

        _names = list(sim.physics.named.data.qpos.axes[0].names)
        root_joint = next((n for n in _names if n.endswith('/') and 'dust' not in n), None)

        dist_history = []
        reached_food = False
        crash_step   = None

        for step in range(N_STEPS):
            odor       = obs['odor_intensity'][0]
            left_odor  = float(odor[0] + odor[2]) / 2.0
            right_odor = float(odor[1] + odor[3]) / 2.0
            total_odor = left_odor + right_odor
            lr_asym    = (right_odor - left_odor) / (total_odor + 1e-9)
            odor_turn  = float(np.tanh(lr_asym * 20.0) * ODOR_TURN_K)

            ctrl = np.array([
                float(np.clip(WALK_AMP + odor_turn, 0.1, 1.0)),
                float(np.clip(WALK_AMP - odor_turn, 0.1, 1.0)),
            ])

            try:
                for _ in range(microsteps):
                    obs, _, _, _, _ = sim.step(ctrl)
            except Exception as e:
                crash_step = step + 1
                print(f"    CRASH at decision step {crash_step}: {type(e).__name__}")
                break

            qpos = sim.physics.named.data.qpos[root_joint]
            pos  = np.array([float(qpos[0]), float(qpos[1])])
            dist = float(np.linalg.norm(pos - FOOD_POS[:2]))
            dist_history.append(dist)

            if dist < FEED_DIST:
                reached_food = True
                print(f"    Reached food at step {step+1}!")
                break

        if not crash_step and not reached_food:
            final = dist_history[-1]
            min_d = min(dist_history)
            print(f"    Did not reach food. steps={len(dist_history)}  min={min_d:.2f}mm  final={final:.2f}mm")

        return reached_food, crash_step, dist_history

    except Exception as e:
        print(f"    SETUP CRASH: {type(e).__name__}: {e}")
        return False, 0, []


# ---------------------------------------------------------------------------
print("=" * 65)
print("Solid wall stability test")
print("Wall: x=14mm, y=6..18, z=2..10  contype=1 (solid)")
print("=" * 65)

results = {}

# Fix 1: disable adhesion (dt=1e-4, normal microsteps)
reached, crash, hist = run_trial(
    "Fix 1: no adhesion", timestep=1e-4, adhesion=False,
    soft_wall=False, microsteps=250)
results['no_adhesion'] = (reached, crash)

# Fix 2: soft contact solimp/solref (adhesion on)
reached, crash, hist = run_trial(
    "Fix 2: soft solimp/solref", timestep=1e-4, adhesion=True,
    soft_wall=True, microsteps=250)
results['soft_contact'] = (reached, crash)

# Fix 3: half timestep (adhesion on, no softening)
reached, crash, hist = run_trial(
    "Fix 3: dt=5e-5 (half timestep)", timestep=5e-5, adhesion=True,
    soft_wall=False, microsteps=500)   # same 25ms per decision
results['half_dt'] = (reached, crash)

# Fix 4: all three combined
reached, crash, hist = run_trial(
    "Fix 4: no adhesion + soft + dt=5e-5", timestep=5e-5, adhesion=False,
    soft_wall=True, microsteps=500)
results['all_combined'] = (reached, crash)

# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
for name, (reached, crash) in results.items():
    status = "OK - reached food" if reached else (f"CRASH at step {crash}" if crash else "STUCK - no food")
    print(f"  {name:<30}  {status}")

print()
working = [k for k, (r, c) in results.items() if r and not c]
if working:
    print(f"  Working fix(es): {working}")
    print(f"  Recommended: {working[0]}")
else:
    print("  No fix worked — solid wall may not be compatible with this fly model")
    print("  Suggestion: use ghost wall + blocked odor source instead")

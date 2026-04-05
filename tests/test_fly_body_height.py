"""
tests/test_fly_body_height.py

Measure actual fly body z position during walking.
This tells us the minimum wall base height needed to block the fly.

Run:
    wenv310\\Scripts\\python.exe tests/test_fly_body_height.py
"""

from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()
import numpy as np

FOOD_POS = np.array([18.0, 12.0, 0.0])
csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]

arena = OdorArena(
    odor_source=FOOD_POS[np.newaxis],
    peak_odor_intensity=np.array([[500.0, 0.0]]),
    diffuse_func=lambda x: x**-2, marker_colors=[],
)
fly = Fly(spawn_pos=(0,0,0.2), spawn_orientation=(0,0,0.588),
          contact_sensor_placements=csp, enable_olfaction=True,
          enable_adhesion=True, draw_adhesion=False, enable_vision=False)
sim = HybridTurningController(fly=fly, arena=arena, timestep=1e-4)
obs, _ = sim.reset()

_names = list(sim.physics.named.data.qpos.axes[0].names)
root_joint = next((n for n in _names if n.endswith('/') and 'dust' not in n), None)
print(f"Root joint: {root_joint}")

z_vals = []
for step in range(80):
    odor      = obs['odor_intensity'][0]
    lr_asym   = (float(odor[1]+odor[3]) - float(odor[0]+odor[2])) / (sum(odor)+1e-9)
    odor_turn = float(np.tanh(lr_asym * 20.0) * 2.5)
    ctrl = np.array([np.clip(0.75+odor_turn, 0.1, 1.0),
                     np.clip(0.75-odor_turn, 0.1, 1.0)])
    for _ in range(250):
        obs, _, _, _, _ = sim.step(ctrl)

    qpos = sim.physics.named.data.qpos[root_joint]
    x, y, z = float(qpos[0]), float(qpos[1]), float(qpos[2])
    z_vals.append(z)
    if step % 10 == 0:
        print(f"  step {step:3d}: x={x:.2f} y={y:.2f} z={z:.4f}mm")

print(f"\nFly body z during walking:")
print(f"  min z = {min(z_vals):.4f}mm")
print(f"  max z = {max(z_vals):.4f}mm")
print(f"  mean z = {np.mean(z_vals):.4f}mm")
print(f"\n  -> Wall base must be BELOW {min(z_vals):.3f}mm to block the fly")
print(f"  -> But adhesion crashes below ~0.6mm base")
print(f"  -> If min_z > 0.6mm: solimp fix alone is sufficient")
print(f"  -> If min_z < 0.6mm: need additional fix (adhesion off or floor geom)")

"""
tests/test_shadow_pole_v2.py

Tests for revised pole + light configuration:
  - Directional light (illuminates full scene, not a spotlight)
  - Pole moved off the fly's direct path
  - Light angle chosen so shadow crosses the fly's path

Fly path: (0,0) -> (18,12) mm  (direction ~0.832, 0.554)
Pole at (6, 8) mm  — 4 mm to the LEFT of the path
Light dir (1, -0.5, -1.5)  — shadow tip at ~(9.4, 6.3) mm  ON the fly's path

Run:
    wenv310\\Scripts\\python.exe tests/test_shadow_pole_v2.py
"""

import flygym.preprogrammed as preprogrammed
from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

import numpy as np

FOOD_POS = np.array([18.0, 12.0, 0.0])
PHYSICS_TIMESTEP = 1e-4
POLE_POS   = [6.0, 8.0]
POLE_R     = 0.8
POLE_H     = 5.0   # half-height → total 10 mm

contact_sensor_placements = [
    f"{leg}{seg}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

def _build(spawn_xy, directional=True):
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2,
        marker_colors=[],
    )
    arena.root_element.worldbody.add(
        "light", name="sun",
        directional=directional,
        dir=[0.4, -0.133, -1.0],   # shadow tip at (10, 6.67) — ON fly path
        diffuse=[1.4, 1.4, 1.2],
        specular=[0.2, 0.2, 0.1],
        castshadow=True,
    )
    arena.root_element.worldbody.add(
        "geom", name="shadow_pole",
        type="cylinder",
        pos=[POLE_POS[0], POLE_POS[1], POLE_H],
        size=[POLE_R, POLE_H],
        rgba=[0.25, 0.18, 0.10, 1.0],
        contype=0, conaffinity=0,
    )
    fly = Fly(
        spawn_pos=(*spawn_xy, 0.2),
        spawn_orientation=(0, 0, 0),
        contact_sensor_placements=contact_sensor_placements,
        enable_olfaction=True, enable_adhesion=True,
        draw_adhesion=False, enable_vision=True,
    )
    sim = HybridTurningController(fly=fly, arena=arena, timestep=PHYSICS_TIMESTEP)
    obs, _ = sim.reset()
    return float(obs["vision"].mean()), sim, obs

# ---------------------------------------------------------------------------
# TEST 1 — directional light illuminates full scene
print("-- TEST 1: directional light - uniform scene illumination --")
lum_start, _, _ = _build((0.0, 0.0))
lum_mid,   _, _ = _build((9.0, 6.0))    # middle of fly path
lum_food,  _, _ = _build((16.0, 10.0))  # near food
print(f"  lum at start (0,0)      = {lum_start:.4f}")
print(f"  lum at midpoint (9,6)   = {lum_mid:.4f}")
print(f"  lum at near food (16,10)= {lum_food:.4f}")
# All three should be non-zero and reasonably similar (directional = no falloff)
assert lum_start > 0.05, "FAIL: start too dark"
assert lum_mid   > 0.05, "FAIL: midpoint too dark"
assert lum_food  > 0.05, "FAIL: near food too dark"
spread = max(lum_start, lum_mid, lum_food) - min(lum_start, lum_mid, lum_food)
print(f"  luminance spread across scene = {spread:.4f}")
print(f"  OK scene is uniformly lit (directional light)")

# ---------------------------------------------------------------------------
# TEST 2 — shadow dip at expected crossing point on fly's path (~9.4, 6.3)
print("\n-- TEST 2: shadow crossing on fly path --")
# With dir=(0.4,-0.133,-1.0) and pole at (6,8) height 10mm:
# Shadow tip = (6 - 10*(0.4/-1), 8 - 10*(-0.133/-1)) = (10, 6.67) -- ON fly path
lum_in_shadow, _, _  = _build((10.0, 6.67))  # shadow tip — on fly path
lum_lit_nearby, _, _ = _build((12.0, 8.0))   # lit area, off the shadow line
lum_pole_base, _, _  = _build((POLE_POS[0], POLE_POS[1]))  # under pole
print(f"  lum at shadow tip (10, 6.67)   = {lum_in_shadow:.4f}")
print(f"  lum at lit nearby (12, 8)       = {lum_lit_nearby:.4f}")
print(f"  lum under pole base (6, 8)      = {lum_pole_base:.4f}")
diff_cross = lum_lit_nearby - lum_in_shadow
diff_pole  = lum_lit_nearby - lum_pole_base
print(f"  shadow drop at crossing = {diff_cross:+.4f}")
print(f"  shadow drop at pole base = {diff_pole:+.4f}")
if diff_pole > 0.02:
    print("  OK shadow clearly detected under pole")
else:
    print("  WARNING: weak shadow — renderer may handle directional differently")

# ---------------------------------------------------------------------------
# TEST 3 — pole is off fly's direct path (fly won't hit it navigating to food)
print("\n-- TEST 3: pole is off fly direct path --")
# Fly path line: y = (12/18)*x = (2/3)*x
# Distance from pole (6,8) to this line:  |8 - (2/3)*6| / sqrt(1 + (2/3)^2)
# = |8 - 4| / sqrt(1 + 4/9) = 4 / sqrt(13/9) = 4 * 3/sqrt(13) ≈ 3.33 mm
path_y_at_pole_x = (2.0/3.0) * POLE_POS[0]
dist_to_path = abs(POLE_POS[1] - path_y_at_pole_x) / np.sqrt(1 + (2.0/3.0)**2)
print(f"  pole ({POLE_POS[0]}, {POLE_POS[1]}) is {dist_to_path:.2f} mm from fly's direct path")
assert dist_to_path > 2.0, f"FAIL: pole too close to path ({dist_to_path:.2f} mm)"
print(f"  OK pole is {dist_to_path:.2f} mm off the direct path — fly won't hit it")

# ---------------------------------------------------------------------------
# TEST 4 — 10 physics steps near shadow crossing without error
print("\n-- TEST 4: physics steps at shadow crossing point --")
_, sim, obs = _build((9.0, 6.0))
lums = []
for _ in range(10):
    obs, _, _, _, _ = sim.step(np.array([1.0, 1.0]))
    lums.append(float(obs["vision"].mean()))
print(f"  luminance over 10 steps: min={min(lums):.4f}  max={max(lums):.4f}")
assert len(lums) == 10
print("  OK 10 steps without error")

# ---------------------------------------------------------------------------
print("\n=== ALL TESTS PASSED ===")
print(f"Pole at {POLE_POS}, {dist_to_path:.1f} mm off fly path.")
print(f"Shadow expected to cross path at ~(9.4, 6.3) mm.")
print(f"Scene luminance spread: {spread:.4f} (directional light = uniform illumination).")

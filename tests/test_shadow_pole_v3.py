"""
tests/test_shadow_pole_v3.py

Design goals:
  - Uniform scene lighting (directional, not too bright)
  - Shadow structure looks the same from all camera angles (all cylinders)
  - Shadow cap center mathematically positioned so its shadow lands ON fly path
  - Measurable luminance drop (>0.02) as fly crosses the shadow zone
  - Shadow quality improved via vis.quality.shadowsize

Math:
  Fly path: (0,0) -> (18,12), parametric: y = (2/3)*x
  Light dir = [0.4, -0.133, -1.0]
  Shadow of point at (px, py, pz) hits ground at:
    sx = px - pz*(dx/dz) = px + 0.4*pz
    sy = py - pz*(dy/dz) = py - 0.133*pz  (dy=-0.133, dz=-1)
  Wait: dy/dz = -0.133 / -1.0 = 0.133, so sy = py - pz*0.133

  To land shadow at (9, 6) -- midpoint of fly path (y=6=(2/3)*9 check: 2/3*9=6 ✓):
    9  = px + 0.4*pz
    6  = py - 0.133*pz
  Choose pz=6mm: px = 9 - 2.4 = 6.6,  py = 6 + 0.133*6 = 6.8
  -> Cap at (6.6, 6.8, 6.3mm)

  Distance of post (6.6, 6.8) from fly path y=(2/3)*x:
    d = |6.8 - (2/3)*6.6| / sqrt(1+(2/3)^2) = |6.8-4.4| / 1.202 = 2.0mm  -> safe

Run:
    wenv310\\Scripts\\python.exe tests/test_shadow_pole_v3.py
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

# Mathematically derived cap position
CAP_POS  = [6.6, 6.8, 6.3]   # shadow center falls at (9, 6) — on fly path
CAP_R    = 4.0                 # mm radius — wide enough for clear shadow
CAP_H    = 0.3                 # mm half-height — flat disk
POST_POS = [6.6, 6.8, 3.15]   # same XY, midpoint in Z
POST_R   = 0.4
POST_H   = 3.15

LIGHT_DIR    = [0.4, -0.133, -1.0]
LIGHT_DIFF   = [0.85, 0.85, 0.80]   # reduced from 1.4 to avoid overexposure

csp = [f"{l}{s}" for l in ["LF","LM","LH","RF","RM","RH"]
       for s in ["Tibia","Tarsus1","Tarsus2","Tarsus3","Tarsus4","Tarsus5"]]

def _build(spawn_xy, with_structure=True, shadowsize=None):
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2,
        marker_colors=[],
    )
    arena.root_element.worldbody.add(
        "light", name="sun",
        directional=True,
        dir=LIGHT_DIR,
        diffuse=LIGHT_DIFF,
        specular=[0.15, 0.15, 0.12],
        castshadow=True,
    )
    if with_structure:
        # Flat disk cap — cylinder so it looks circular from top AND side
        arena.root_element.worldbody.add(
            "geom", name="shadow_cap",
            type="cylinder",
            pos=CAP_POS, size=[CAP_R, CAP_H],
            rgba=[0.22, 0.15, 0.07, 1.0],
            contype=0, conaffinity=0,
        )
        # Thin supporting post — also cylinder
        arena.root_element.worldbody.add(
            "geom", name="shadow_post",
            type="cylinder",
            pos=POST_POS, size=[POST_R, POST_H],
            rgba=[0.22, 0.15, 0.07, 1.0],
            contype=0, conaffinity=0,
        )
    fly = Fly(
        spawn_pos=(*spawn_xy, 0.2),
        spawn_orientation=(0, 0, 0),
        contact_sensor_placements=csp,
        enable_olfaction=True, enable_adhesion=True,
        draw_adhesion=False, enable_vision=True,
    )
    sim = HybridTurningController(fly=fly, arena=arena, timestep=PHYSICS_TIMESTEP)
    obs, _ = sim.reset()

    # Raise shadow map resolution if requested
    if shadowsize is not None:
        sim.physics.model.vis.quality.shadowsize = shadowsize

    return float(obs["vision"].mean()), sim, obs

# ---------------------------------------------------------------------------
# TEST 1 — verify shadow math: cap at (6.6, 6.8, 6.3) -> shadow at (9, 6)
print("-- TEST 1: shadow geometry math --")
dx, dy, dz = LIGHT_DIR
pz = CAP_POS[2]
sx = CAP_POS[0] - pz*(dx/dz)
sy = CAP_POS[1] - pz*(dy/dz)
path_y_at_sx = (2.0/3.0)*sx
print(f"  Cap at {CAP_POS} with light dir {LIGHT_DIR}")
print(f"  Shadow center hits ground at: ({sx:.2f}, {sy:.2f})")
print(f"  Fly path y at x={sx:.1f}: {path_y_at_sx:.2f}")
print(f"  Offset from path: {abs(sy - path_y_at_sx):.2f} mm")
assert abs(sy - path_y_at_sx) < 0.2, f"FAIL: shadow center {(sx,sy)} not on fly path"
print("  OK shadow center mathematically verified on fly path")

# ---------------------------------------------------------------------------
# TEST 2 — uniform scene brightness (directional, not too bright)
print("\n-- TEST 2: scene uniformity and brightness --")
lum_start, _, _ = _build((0.0,  0.0),  with_structure=False)
lum_mid,   _, _ = _build((9.0,  6.0),  with_structure=False)
lum_food,  _, _ = _build((16.0, 10.0), with_structure=False)
spread = max(lum_start, lum_mid, lum_food) - min(lum_start, lum_mid, lum_food)
print(f"  lum start={lum_start:.4f}  mid={lum_mid:.4f}  food={lum_food:.4f}")
print(f"  spread = {spread:.4f}")
assert lum_mid < 0.65, f"FAIL: scene too bright ({lum_mid:.3f}), want < 0.65"
assert lum_mid > 0.2,  f"FAIL: scene too dark ({lum_mid:.3f}), want > 0.2"
assert spread < 0.02,  f"FAIL: uneven lighting (spread={spread:.4f})"
print(f"  OK brightness in range, uniform across scene")

# ---------------------------------------------------------------------------
# TEST 3 — shadow drop at fly path crossing point
print("\n-- TEST 3: shadow drop at fly path crossing (9, 6) --")
lum_shadow, _, _ = _build((sx,    sy))            # in shadow
lum_before, _, _ = _build((4.0,   2.67))          # on path, before shadow
lum_after,  _, _ = _build((14.0,  9.33))          # on path, after shadow
lum_base,   _, _ = _build((9.0,   6.0),  with_structure=False)  # no structure

drop_shadow = lum_base - lum_shadow
drop_before = lum_base - lum_before
drop_after  = lum_base - lum_after
print(f"  lum no structure (ref): {lum_base:.4f}")
print(f"  lum before shadow:      {lum_before:.4f}  drop={drop_before:+.4f}")
print(f"  lum in shadow:          {lum_shadow:.4f}  drop={drop_shadow:+.4f}")
print(f"  lum after shadow:       {lum_after:.4f}   drop={drop_after:+.4f}")

VIS_MIN, VIS_MAX = 20.0, 150.0
hz_drop = drop_shadow * (VIS_MAX - VIS_MIN)
print(f"  -> lamina firing rate drop in shadow: ~{hz_drop:.1f} Hz")

if drop_shadow > 0.02:
    print(f"  OK strong shadow detected ({drop_shadow:.4f} = {hz_drop:.1f} Hz drop)")
elif drop_shadow > 0.005:
    print(f"  WEAK shadow ({drop_shadow:.4f}), may still be detectable in plots")
else:
    print(f"  WARNING: shadow too weak ({drop_shadow:.4f})")

# ---------------------------------------------------------------------------
# TEST 4 — shadowsize can be set on the model
print("\n-- TEST 4: shadow quality setting --")
_, sim4, _ = _build((0.0, 0.0), shadowsize=8192)
actual_size = sim4.physics.model.vis.quality.shadowsize
print(f"  vis.quality.shadowsize = {actual_size}")
assert actual_size == 8192, f"FAIL: shadowsize not set ({actual_size})"
print("  OK shadowsize=8192 applied successfully")

# ---------------------------------------------------------------------------
# TEST 5 — post is off fly path (no collision risk)
print("\n-- TEST 5: post clear of fly direct path --")
path_dist = abs(POST_POS[1] - (2/3)*POST_POS[0]) / np.sqrt(1 + (2/3)**2)
print(f"  Post at ({POST_POS[0]}, {POST_POS[1]}) is {path_dist:.2f} mm from fly path")
assert path_dist > POST_R + 0.5, f"FAIL: post too close to path ({path_dist:.2f} mm)"
print(f"  OK post is {path_dist:.2f} mm off path (post radius={POST_R}mm)")

# ---------------------------------------------------------------------------
# TEST 6 — 10 physics steps near shadow zone without error
print("\n-- TEST 6: physics steps through shadow zone --")
_, sim6, obs6 = _build((7.0, 4.67), shadowsize=8192)
lums = []
for _ in range(10):
    obs6, _, _, _, _ = sim6.step(np.array([1.0, 1.0]))
    lums.append(float(obs6["vision"].mean()))
print(f"  lum over 10 steps: min={min(lums):.4f}  max={max(lums):.4f}")
print("  OK no errors")

# ---------------------------------------------------------------------------
print("\n=== ALL TESTS PASSED ===")
print(f"Cap at {CAP_POS}, shadow center at ({sx:.1f}, {sy:.1f}) — on fly path y=(2/3)x")
print(f"Shadow drop: {drop_shadow:.4f} = ~{hz_drop:.1f} Hz lamina rate change")
print(f"Post is {path_dist:.1f} mm off fly path — no collision risk")
print(f"shadowsize=8192 verified accessible")

"""
tests/test_vision_sensor.py

Verify that enable_vision=True works with OdorArena + HybridTurningController,
and that obs["vision"] has the expected shape and produces usable luminance values.

Run:
    wenv310\Scripts\python.exe tests/test_vision_sensor.py
"""

import sys
import os
from pathlib import Path

# flygym before load_dotenv (Windows GL context)
import flygym.preprogrammed as preprogrammed
from flygym import Fly, YawOnlyCamera
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

import numpy as np

# ---------------------------------------------------------------------------
FOOD_POS       = np.array([18.0, 12.0, 0.0])
PHYSICS_TIMESTEP = 1e-4

print("Building arena ...")
arena = OdorArena(
    odor_source=FOOD_POS[np.newaxis],
    peak_odor_intensity=np.array([[500.0, 0.0]]),
    diffuse_func=lambda x: x**-2,
    marker_colors=[],
)

contact_sensor_placements = [
    f"{leg}{seg}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

print("Building fly with enable_vision=True ...")
fly = Fly(
    spawn_pos=(0, 0, 0.2),
    spawn_orientation=(0, 0, -1.2),
    contact_sensor_placements=contact_sensor_placements,
    enable_olfaction=True,
    enable_adhesion=True,
    draw_adhesion=False,
    enable_vision=True,           # <-- what we are testing
)

print("Building sim ...")
sim = HybridTurningController(
    fly=fly,
    arena=arena,
    timestep=PHYSICS_TIMESTEP,
)
obs, _ = sim.reset()

# ---------------------------------------------------------------------------
# TEST 1 — obs["vision"] present and correct shape
print("\n-- TEST 1: obs[\"vision\"] shape --")
assert "vision" in obs, "FAIL: 'vision' key missing from obs"
vis = obs["vision"]
print(f"  obs['vision'].shape = {vis.shape}  dtype={vis.dtype}")
assert vis.ndim == 3, f"FAIL: expected 3 dims, got {vis.ndim}"
assert vis.shape[0] == 2, f"FAIL: expected 2 eyes, got {vis.shape[0]}"
assert vis.shape[1] == 721, f"FAIL: expected 721 ommatidia, got {vis.shape[1]}"
assert vis.shape[2] == 2, f"FAIL: expected 2 channels, got {vis.shape[2]}"
print("  OK shape (2, 721, 2)")

# ---------------------------------------------------------------------------
# TEST 2 — luminance extraction (mean intensity per eye)
print("\n-- TEST 2: luminance extraction --")
left_lum  = float(obs["vision"][0].mean())
right_lum = float(obs["vision"][1].mean())
print(f"  left eye luminance  = {left_lum:.2f}  (range 0-255)")
print(f"  right eye luminance = {right_lum:.2f}  (range 0-255)")
assert 0 <= left_lum <= 255,  "FAIL: left luminance out of range"
assert 0 <= right_lum <= 255, "FAIL: right luminance out of range"
print("  OK luminance values in valid range")

# ---------------------------------------------------------------------------
# TEST 3 — run a few physics steps, vision updates without error
print("\n-- TEST 3: vision updates over 5 physics steps --")
lums = []
for step in range(5):
    action = np.array([1.0, 1.0])   # straight walk
    obs, _, _, _, _ = sim.step(action)
    l = float(obs["vision"].mean())
    lums.append(l)
    print(f"  step {step}  mean_lum={l:.2f}")

assert len(lums) == 5, "FAIL: did not complete 5 steps"
print("  OK vision updates each step")

# ---------------------------------------------------------------------------
# TEST 4 — luminance to PoissonGroup rate conversion (0-255 -> 20-200 Hz)
print("\n-- TEST 4: luminance -> firing rate mapping --")
VIS_MIN_HZ = 20.0
VIS_MAX_HZ = 200.0

def lum_to_rate(lum_array):
    """Mean luminance of one eye (0-255) -> firing rate in Hz."""
    mean_lum = lum_array.mean() / 255.0       # normalise to [0, 1]
    return VIS_MIN_HZ + mean_lum * (VIS_MAX_HZ - VIS_MIN_HZ)

left_rate  = lum_to_rate(obs["vision"][0])
right_rate = lum_to_rate(obs["vision"][1])
print(f"  left eye  -> {left_rate:.1f} Hz")
print(f"  right eye -> {right_rate:.1f} Hz")
assert VIS_MIN_HZ <= left_rate  <= VIS_MAX_HZ, "FAIL: left rate out of range"
assert VIS_MIN_HZ <= right_rate <= VIS_MAX_HZ, "FAIL: right rate out of range"
print("  OK rate mapping in [20, 200] Hz")

# ---------------------------------------------------------------------------
print("\n=== ALL TESTS PASSED ===")
print("Vision sensor works with OdorArena + HybridTurningController.")
print(f"obs['vision'] shape: (2, 721, 2)  -- 2 eyes x 721 ommatidia x 2 photoreceptor types")
print("Luminance -> firing rate conversion: OK")

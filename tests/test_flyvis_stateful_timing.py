"""
tests/test_flyvis_stateful_timing.py

Test flyvis stateful forward() as a faster alternative to simulate(8 frames).

The simulate() call runs steady_state(1s, dt) every step = 40 grey frames overhead.
Stateful forward() keeps recurrent state across steps -> just 1 frame per step.

Approach:
  1. Compute steady state once at init
  2. Each step: stimulus.zero + add_input + forward(stimulus(), dt, state)
  3. Extract T5 activity from state.nodes.activity using layer_index

Run:
    wenv310\\Scripts\\python.exe tests/test_flyvis_stateful_timing.py
"""

from flygym import Fly  # must come before load_dotenv on Windows
from dotenv import load_dotenv
load_dotenv()

import importlib.util, sys, time
import numpy as np
import torch

def _load_retina_mapper():
    spec = importlib.util.spec_from_file_location(
        'flygym.vision.retina',
        'wenv310/lib/site-packages/flygym/vision/retina.py')
    retina_mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('flygym.vision.retina', retina_mod)
    sys.modules.setdefault('flygym.vision', retina_mod)
    spec.loader.exec_module(retina_mod)
    spec2 = importlib.util.spec_from_file_location(
        'vision_network',
        'wenv310/lib/site-packages/flygym/examples/vision/vision_network.py')
    vn_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(vn_mod)
    return vn_mod.RetinaMapper()

import flyvis
from flyvis import NetworkView
from flyvis.utils.nn_utils import simulation

print("Loading flyvis network...")
nv      = NetworkView(flyvis.results_dir / 'flow/0000/000')
network = nv.init_network(checkpoint='best')
rm      = _load_retina_mapper()
print("Network loaded.\n")

t5a_idx = network.stimulus.layer_index['T5a']
t5b_idx = network.stimulus.layer_index['T5b']
print(f"T5a cells: {len(t5a_idx)}   T5b cells: {len(t5b_idx)}")
print(f"Total network cells: {network.stimulus.n_nodes}")

DT       = 25e-3
N_WARMUP = 3
N_TIMED  = 15

# Synthetic frames: grey with dark patch on left to produce asymmetric T5
def make_frame(dark_left=True):
    f = np.ones((2, 721), dtype=np.float32) * 0.5
    if dark_left:
        f[0, :80] = 0.1  # dark edge in left eye
    return f

# ---- Method A: simulate(8 frames) per step ----
print("\n=== Method A: simulate(8 frames) per step ===")
buf_A = np.ones((2, 8, 721), dtype=np.float32) * 0.5

times_A = []
for i in range(N_WARMUP + N_TIMED):
    frame = make_frame(dark_left=(i >= N_WARMUP))
    mapped = rm.flygym_to_flyvis(frame)  # (2, 721)
    buf_A = np.roll(buf_A, -1, axis=1)
    buf_A[:, -1, :] = mapped
    movie = torch.tensor(buf_A, dtype=torch.float32).unsqueeze(2)  # (2,8,1,721)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = network.simulate(movie, dt=DT, as_layer_activity=True)
    dt_ms = (time.perf_counter() - t0) * 1000

    t5_l = float((out['T5a'][0,-1].abs() + out['T5b'][0,-1].abs()).mean())
    t5_r = float((out['T5a'][1,-1].abs() + out['T5b'][1,-1].abs()).mean())

    if i >= N_WARMUP:
        times_A.append(dt_ms)
        print(f"  step {i-N_WARMUP+1:2d}: {dt_ms:7.1f}ms  T5_L={t5_l:.4f} T5_R={t5_r:.4f}")
    else:
        print(f"  [warmup {i+1}] {dt_ms:.0f}ms")

print(f"\n  A median: {np.median(times_A):.1f}ms  mean: {np.mean(times_A):.1f}ms")
print(f"  A 400-step estimate: {np.median(times_A)*400/60000:.1f} min")

# ---- Method B: stateful forward(), 1 frame per step ----
print("\n=== Method B: stateful forward(), 1 frame per step ===")

print("  Computing steady state (1s of grey = 40 frames)...")
t_init = time.perf_counter()
network.eval()
for p in network.parameters():
    p.requires_grad = False

grey_t = torch.ones((2, 1, 1, 721), dtype=torch.float32) * 0.5
with torch.no_grad():
    network.stimulus.zero(2, 1)
    network.stimulus.add_input(grey_t)
    state = network.forward(network.stimulus(), DT, state=None, as_states=True)[-1]
    for _ in range(39):
        network.stimulus.zero(2, 1)
        network.stimulus.add_input(grey_t)
        state = network.forward(network.stimulus(), DT, state=state, as_states=True)[-1]
print(f"  Steady state done in {(time.perf_counter()-t_init)*1000:.0f}ms")

times_B = []
for i in range(N_WARMUP + N_TIMED):
    frame = make_frame(dark_left=(i >= N_WARMUP))
    mapped = rm.flygym_to_flyvis(frame)  # (2, 721)
    frame_t = torch.tensor(mapped, dtype=torch.float32).unsqueeze(1).unsqueeze(2)  # (2,1,1,721)

    t0 = time.perf_counter()
    with torch.no_grad():
        network.stimulus.zero(2, 1)
        network.stimulus.add_input(frame_t)
        new_states = network.forward(network.stimulus(), DT, state=state, as_states=True)
        state = new_states[-1]
        act = state.nodes.activity  # (2, n_cells)
        t5_l = float(act[0, t5a_idx].abs().mean() + act[0, t5b_idx].abs().mean())
        t5_r = float(act[1, t5a_idx].abs().mean() + act[1, t5b_idx].abs().mean())
    dt_ms = (time.perf_counter() - t0) * 1000

    if i >= N_WARMUP:
        times_B.append(dt_ms)
        print(f"  step {i-N_WARMUP+1:2d}: {dt_ms:7.1f}ms  T5_L={t5_l:.4f} T5_R={t5_r:.4f}")
    else:
        print(f"  [warmup {i+1}] {dt_ms:.0f}ms")

print(f"\n  B median: {np.median(times_B):.1f}ms  mean: {np.mean(times_B):.1f}ms")
print(f"  B 400-step estimate: {np.median(times_B)*400/60000:.1f} min")

print(f"\n=== SUMMARY ===")
print(f"  Method A (simulate 8 frames): {np.median(times_A):.0f}ms/step  -> {np.median(times_A)*400/60000:.1f} min extra")
print(f"  Method B (stateful forward):  {np.median(times_B):.0f}ms/step  -> {np.median(times_B)*400/60000:.1f} min extra")
if np.median(times_B) > 0:
    print(f"  Speedup: {np.median(times_A)/np.median(times_B):.1f}x")
print(f"  (Main sim cost: ~20-30s/step = 130-200 min for 400 steps)")

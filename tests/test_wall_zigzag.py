"""
tests/test_wall_zigzag.py

Zigzag / tunnel layout:

  Wall 1  x=8,  y=0..10  gap at top    (y >= 10)
  Wall 2  x=14, y=6..16  gap at bottom (y <= 6)
  Tunnel  x=8..14, y=6..10

  Fly goes northeast -> north over Wall 1 gap -> east ->
  south through tunnel -> east to food at (20, 2).

3 conditions:
  A - no walls (baseline)
  B - both walls + flyvis T5
  C - both walls, odor only

Run:
    wenv310\Scripts\python.exe tests/test_wall_zigzag.py
"""

from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

import importlib.util, sys
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


# ── Constants ─────────────────────────────────────────────────────────────────
FOOD_POS         = np.array([20.0, 2.0, 0.0])
SPAWN_POS        = (0, 0, 0.2)
SPAWN_HEADING    = (0, 0, 0.588)
PHYSICS_TIMESTEP = 1e-4
WALK_AMP         = 0.75
ODOR_TURN_K      = 2.5
FEED_DIST        = 1.2
PHYSICS_STEPS    = 250
N_STEPS          = 500          # more steps - longer route

FLYVIS_T5_GAIN  = 0.5
FLYVIS_DECAY    = 0.5
FLYVIS_BIAS_MAX = 0.15
FLYVIS_DT       = 25e-3

# Wall 1: x=8, center y=5, half=5 -> spans y=0..10, gap at y>=10
WALL_1 = {
    'name': 'wall1',
    'type': 'box', 'pos': [8.0, 5.0, 3.0], 'size': [0.3, 5.0, 4.0],
    'rgba': [0.15, 0.10, 0.05, 1.0], 'contype': 1, 'conaffinity': 1,
    'solimp': '0.9 0.999 0.001 0.5 2', 'solref': '0.02 1',
}

# Wall 2: x=14, center y=11, half=5 -> spans y=6..16, gap at y<=6
WALL_2 = {
    'name': 'wall2',
    'type': 'box', 'pos': [14.0, 11.0, 3.0], 'size': [0.3, 5.0, 4.0],
    'rgba': [0.15, 0.10, 0.05, 1.0], 'contype': 1, 'conaffinity': 1,
    'solimp': '0.9 0.999 0.001 0.5 2', 'solref': '0.02 1',
}

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]


def run_scenario(use_walls, use_flyvis, network, rm, t5a_idx, t5b_idx,
                 fv_state_init, label):
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2, marker_colors=[],
    )
    if use_walls:
        arena.root_element.worldbody.add('geom', **WALL_1)
        arena.root_element.worldbody.add('geom', **WALL_2)

    fly = Fly(spawn_pos=SPAWN_POS, spawn_orientation=SPAWN_HEADING,
              contact_sensor_placements=csp, enable_olfaction=True,
              enable_adhesion=True, draw_adhesion=False, enable_vision=True)
    sim = HybridTurningController(fly=fly, arena=arena, timestep=PHYSICS_TIMESTEP)
    obs, _ = sim.reset()
    _names = list(sim.physics.named.data.qpos.axes[0].names)
    root_joint = next((n for n in _names if n.endswith('/') and 'dust' not in n), None)

    fv_state     = fv_state_init
    loom_persist = 0.0
    x_hist = []; y_hist = []; dist_hist = []
    t5l_hist = []; t5r_hist = []; bias_hist = []
    reached = False

    for step in range(N_STEPS):
        odor     = obs['odor_intensity'][0]
        lr_asym  = (float(odor[1]+odor[3]) - float(odor[0]+odor[2])) / (sum(odor)+1e-9)
        odor_turn = float(np.tanh(lr_asym * 20.0) * ODOR_TURN_K)

        t5_l = t5_r = 0.0
        vis = obs.get('vision')
        if vis is not None and use_flyvis:
            vis_gray   = vis.max(axis=2).astype(np.float32)
            vis_mapped = rm.flygym_to_flyvis(vis_gray)
            frame_t    = torch.tensor(vis_mapped, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
            with torch.no_grad():
                network.stimulus.zero(2, 1)
                network.stimulus.add_input(frame_t)
                fv_states = network.forward(network.stimulus(), FLYVIS_DT, state=fv_state, as_states=True)
                fv_state  = fv_states[-1]
                act = fv_state.nodes.activity
                t5_l = float(act[0, t5a_idx].abs().mean() + act[0, t5b_idx].abs().mean())
                t5_r = float(act[1, t5a_idx].abs().mean() + act[1, t5b_idx].abs().mean())
            loom_new     = -FLYVIS_T5_GAIN * (t5_l - t5_r)
            loom_persist = float(np.clip(
                loom_persist * FLYVIS_DECAY + loom_new * (1 - FLYVIS_DECAY),
                -FLYVIS_BIAS_MAX, FLYVIS_BIAS_MAX))

        turn_bias = odor_turn + loom_persist
        ctrl = np.array([np.clip(WALK_AMP+turn_bias, 0.1, 1.0),
                         np.clip(WALK_AMP-turn_bias, 0.1, 1.0)])
        try:
            for _ in range(PHYSICS_STEPS):
                obs, _, _, _, _ = sim.step(ctrl)
        except Exception as e:
            print(f"  [{label}] CRASH step {step+1}: {e}")
            break

        qpos = sim.physics.named.data.qpos[root_joint]
        pos  = np.array([float(qpos[0]), float(qpos[1])])
        dist = float(np.linalg.norm(pos - FOOD_POS[:2]))
        x_hist.append(pos[0]); y_hist.append(pos[1])
        dist_hist.append(dist)
        t5l_hist.append(t5_l); t5r_hist.append(t5_r)
        bias_hist.append(loom_persist)
        if dist < FEED_DIST:
            reached = True
            break

    x = np.array(x_hist); y = np.array(y_hist)

    # Wall 1 crossing: x=8
    cross1 = np.where((x[:-1] < 8) & (x[1:] >= 8))[0]
    # Wall 2 crossing: x=14
    cross2 = np.where((x[:-1] < 14) & (x[1:] >= 14))[0]

    min_d    = min(dist_hist) if dist_hist else 999
    max_y    = max(y_hist) if y_hist else 0

    print(f"\n  [{label}]")
    status = f"Reached food at step {len(dist_hist)}" if reached else f"Did NOT reach food  min={round(min_d,1)}mm"
    print(f"    {status}")
    print(f"    Max y reached: {max_y:.2f}mm")

    if use_walls:
        if len(cross1):
            ci = cross1[0]
            y_c1 = y[ci]
            tag = "VALID (y>=10) OK" if y_c1 >= 10 else f"BLOCKED (y<10) FAIL  y={y_c1:.2f}"
            print(f"    Wall 1 crossed step {ci}: y={y_c1:.2f}  -> {tag}")
        else:
            print(f"    Wall 1: never crossed x=8")

        if len(cross2):
            ci = cross2[0]
            y_c2 = y[ci]
            tag = "VALID (y<=6) OK" if y_c2 <= 6 else f"WRONG SIDE (y>6) FAIL  y={y_c2:.2f}"
            print(f"    Wall 2 crossed step {ci}: y={y_c2:.2f}  -> {tag}")
        else:
            print(f"    Wall 2: never crossed x=14")

        # Show y range in tunnel zone
        tunnel = (x >= 8) & (x <= 14)
        if tunnel.any():
            print(f"    y in tunnel zone (x=8..14): {y[tunnel].min():.2f} .. {y[tunnel].max():.2f}")

    return reached


# ── Load flyvis once ───────────────────────────────────────────────────────────
print("Loading flyvis network...")
import flyvis
from flyvis import NetworkView
nv      = NetworkView(flyvis.results_dir / 'flow/0000/000')
network = nv.init_network(checkpoint='best')
rm      = _load_retina_mapper()
t5a_idx = network.stimulus.layer_index['T5a']
t5b_idx = network.stimulus.layer_index['T5b']

network.eval()
for p in network.parameters():
    p.requires_grad = False
grey_t = torch.ones((2, 1, 1, 721), dtype=torch.float32) * 0.5
print("Computing steady state...")
with torch.no_grad():
    network.stimulus.zero(2, 1)
    network.stimulus.add_input(grey_t)
    fv_state_init = network.forward(network.stimulus(), FLYVIS_DT, state=None, as_states=True)[-1]
    for _ in range(39):
        network.stimulus.zero(2, 1)
        network.stimulus.add_input(grey_t)
        fv_state_init = network.forward(network.stimulus(), FLYVIS_DT, state=fv_state_init, as_states=True)[-1]
print("Ready.\n")

print("=" * 65)
print("ZIGZAG TEST  -  tunnel layout")
print("  Wall 1: x=8,  y=0..10  gap at y>=10")
print("  Wall 2: x=14, y=6..16  gap at y<=6")
print("  Food:   (20, 2)")
print("=" * 65)

r_a = run_scenario(False, False, network, rm, t5a_idx, t5b_idx, fv_state_init,
                   "A: no walls")
r_b = run_scenario(True,  True,  network, rm, t5a_idx, t5b_idx, fv_state_init,
                   "B: both walls + flyvis T5")
r_c = run_scenario(True,  False, network, rm, t5a_idx, t5b_idx, fv_state_init,
                   "C: both walls, odor only")

print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
ok = lambda b: "OK" if b else "FAIL"
print(f"  A (no walls):        {ok(r_a)}")
print(f"  B (walls + flyvis):  {ok(r_b)}")
print(f"  C (walls odor-only): {ok(r_c)}")

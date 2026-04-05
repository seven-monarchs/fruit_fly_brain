"""
tests/test_wall_v40_config.py

Tests the wall config matching v40 main simulation exactly:
  - Fly start: (0,0) heading=-1.2 rad  (same as main sim)
  - Food: (18,12)
  - Wall: x=14, y=4..20, pos_z=4.5, half_height=4 -> base at z=0.5mm (new fix)
  - flyvis stateful forward() (same as main sim, NOT simulate())

Conditions:
  A) No wall — what is the natural path with this heading?
  B) Wall (new z=0.5 base) + flyvis reflex
  C) Wall (new z=0.5 base) + no reflex (odor only)

Reports:
  - Did fly walk under wall or get blocked?
  - y position when crossing x=14
  - T5 signal strength at wall
  - Steps to food

Run:
    wenv310\\Scripts\\python.exe tests/test_wall_v40_config.py
"""

from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
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

# ── Exact main sim parameters ──────────────────────────────────────────────────
FOOD_POS         = np.array([18.0, 12.0, 0.0])
SPAWN_POS        = (0, 0, 0.2)
SPAWN_HEADING    = (0, 0, -1.2)   # main sim heading
PHYSICS_TIMESTEP = 1e-4
WALK_AMP         = 0.75
ODOR_TURN_K      = 2.5
FEED_DIST        = 1.2
PHYSICS_STEPS    = 250
N_STEPS          = 350

FLYVIS_T5_GAIN  = 4.0
FLYVIS_DECAY    = 0.5
FLYVIS_BIAS_MAX = 2.0
FLYVIS_DT       = 25e-3

# ── Wall: new config with base at z=0.5mm ──────────────────────────────────────
WALL = {
    'type': 'box', 'pos': [14.0, 12.0, 4.65], 'size': [0.3, 8.0, 4.0],
    'rgba': [0.15, 0.10, 0.05, 1.0], 'contype': 1, 'conaffinity': 1,
    'solimp': '0.9 0.999 0.001 0.5 2', 'solref': '0.02 1',
}
# Wall spans y=4..20, base z=0.5, top z=8.5
# adhesion disabled on fly to prevent BADQACC at this base height
# Gap: y=0..4

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]


def run_scenario(use_wall, use_flyvis, network, rm, t5a_idx, t5b_idx, fv_state_init, label):
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2, marker_colors=[],
    )
    if use_wall:
        arena.root_element.worldbody.add('geom', **WALL)

    fly = Fly(spawn_pos=SPAWN_POS, spawn_orientation=SPAWN_HEADING,
              contact_sensor_placements=csp, enable_olfaction=True,
              enable_adhesion=False, draw_adhesion=False, enable_vision=True)
    sim = HybridTurningController(fly=fly, arena=arena, timestep=PHYSICS_TIMESTEP)
    obs, _ = sim.reset()
    _names = list(sim.physics.named.data.qpos.axes[0].names)
    root_joint = next((n for n in _names if n.endswith('/') and 'dust' not in n), None)

    fv_state    = fv_state_init
    loom_persist = 0.0
    x_hist = []; y_hist = []; dist_hist = []
    t5l_hist = []; t5r_hist = []; bias_hist = []
    reached = False
    crashed = False

    for step in range(N_STEPS):
        odor      = obs['odor_intensity'][0]
        lr_asym   = (float(odor[1]+odor[3]) - float(odor[0]+odor[2])) / (sum(odor)+1e-9)
        odor_turn = float(np.tanh(lr_asym * 20.0) * ODOR_TURN_K)

        t5_l = t5_r = 0.0
        vis = obs.get('vision')
        if vis is not None and use_flyvis:
            vis_gray  = vis.max(axis=2).astype(np.float32)
            vis_mapped = rm.flygym_to_flyvis(vis_gray)
            frame_t   = torch.tensor(vis_mapped, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
            with torch.no_grad():
                network.stimulus.zero(2, 1)
                network.stimulus.add_input(frame_t)
                fv_states = network.forward(network.stimulus(), FLYVIS_DT, state=fv_state, as_states=True)
                fv_state  = fv_states[-1]
                act = fv_state.nodes.activity
                t5_l = float(act[0, t5a_idx].abs().mean() + act[0, t5b_idx].abs().mean())
                t5_r = float(act[1, t5a_idx].abs().mean() + act[1, t5b_idx].abs().mean())
            t5_asym    = t5_l - t5_r
            loom_new   = -FLYVIS_T5_GAIN * t5_asym
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
            crashed = True
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

    # Wall crossing analysis
    x = np.array(x_hist); y = np.array(y_hist)
    crossing = np.where((x[:-1] < 14) & (x[1:] >= 14))[0]

    min_d = min(dist_hist) if dist_hist else 999
    print(f"\n  [{label}]")
    print(f"    {'Reached food at step '+str(len(dist_hist)) if reached else 'Did NOT reach food  min='+str(round(min_d,1))+'mm'}")
    if use_wall and len(crossing):
        ci = crossing[0]
        gap = y[ci] < 4.0
        print(f"    x=14 crossed at step {ci}: y={y[ci]:.2f}  -> {'THROUGH GAP (y<4) OK' if gap else 'THROUGH WALL BODY (y>=4) -- walked under or clipped!'}")
        if len(t5l_hist) > ci:
            print(f"    T5 at crossing: L={t5l_hist[ci]:.3f} R={t5r_hist[ci]:.3f} bias={bias_hist[ci]:.3f}")
    elif use_wall:
        print(f"    Never crossed x=14 (blocked by wall or didn't reach it)")
    if use_wall:
        wall_zone = (x > 11) & (x < 14)
        if wall_zone.any():
            print(f"    y range approaching wall (x=11..14): {y[wall_zone].min():.2f} .. {y[wall_zone].max():.2f}")
            max_t5r = max(t5r_hist[i] for i in range(len(x_hist)) if wall_zone[i]) if use_flyvis else 0
            print(f"    Max T5_R near wall: {max_t5r:.3f}  (wall to fly's right -> T5_R should spike)")

    return x_hist, y_hist, dist_hist, reached, crashed


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
print("Wall config test — matching v40 main sim exactly")
print(f"Fly: (0,0) heading=-1.2 rad   Food: (18,12)")
print(f"Wall: x=14, y=4..20, pos_z=4.5, base=0.5mm (NEW)")
print("=" * 65)

run_scenario(False, False, network, rm, t5a_idx, t5b_idx, fv_state_init, "A: no wall, no reflex")
run_scenario(True,  True,  network, rm, t5a_idx, t5b_idx, fv_state_init, "B: wall + flyvis reflex")
run_scenario(True,  False, network, rm, t5a_idx, t5b_idx, fv_state_init, "C: wall, no reflex (odor only)")

print("\n" + "=" * 65)
print("If B or C crossed at y>=4: fly walked under wall -> lower z further")
print("If B or C crossed at y<4:  gap used correctly")
print("If never crossed x=14:     wall is blocking (good!)")
print("=" * 65)

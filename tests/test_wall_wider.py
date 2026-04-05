"""
tests/test_wall_wider.py

Tests two approaches to make the wall require a more visible detour:

  Option A — Extended wall (gap at y>=10):
    pos=[14, 5, 3], size=[0.3, 5, 4] -> spans y=0..10
    Fly hits at y~7, needs ~3mm northward travel (vs 2mm current)
    Heading: 0.588 rad (same as current main sim)

  Option B — Lower approach heading (gap stays at y>=9):
    pos=[14, 4.5, 3], size=[0.3, 4.5, 4] -> spans y=0..9 (current wall)
    Heading: 0.3 rad (~17 deg, more eastward)
    Fly hits wall at lower y -> longer detour to reach gap at y=9

Each option tests 3 conditions:
  - no wall baseline
  - wall + flyvis T5 reflex
  - wall, odor only

Run:
    wenv310\\Scripts\\python.exe tests/test_wall_wider.py
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


# ── Shared constants ───────────────────────────────────────────────────────────
FOOD_POS         = np.array([18.0, 12.0, 0.0])
SPAWN_POS        = (0, 0, 0.2)
PHYSICS_TIMESTEP = 1e-4
WALK_AMP         = 0.75
ODOR_TURN_K      = 2.5
FEED_DIST        = 1.2
PHYSICS_STEPS    = 250
N_STEPS          = 400

FLYVIS_T5_GAIN  = 0.5
FLYVIS_DECAY    = 0.5
FLYVIS_BIAS_MAX = 0.15
FLYVIS_DT       = 25e-3

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]

# ── Option A: extended wall, heading unchanged ─────────────────────────────────
# center y=5, half_span=5 -> spans y=0..10, gap at y>=10
# base z=-1mm (same as proven config)
WALL_A = {
    'type': 'box', 'pos': [14.0, 5.0, 3.0], 'size': [0.3, 5.0, 4.0],
    'rgba': [0.15, 0.10, 0.05, 1.0], 'contype': 1, 'conaffinity': 1,
    'solimp': '0.9 0.999 0.001 0.5 2', 'solref': '0.02 1',
}
HEADING_A = (0, 0, 0.588)
GAP_A     = 10.0   # y threshold for valid crossing

# ── Option B: same wall, lower heading ────────────────────────────────────────
# heading=0.3 rad (~17 deg, more eastward) -> fly hits wall at lower y
# wall unchanged: y=0..9, gap at y>=9
WALL_B = {
    'type': 'box', 'pos': [14.0, 4.5, 3.0], 'size': [0.3, 4.5, 4.0],
    'rgba': [0.15, 0.10, 0.05, 1.0], 'contype': 1, 'conaffinity': 1,
    'solimp': '0.9 0.999 0.001 0.5 2', 'solref': '0.02 1',
}
HEADING_B = (0, 0, 0.3)
GAP_B     = 9.0


def run_scenario(use_wall, use_flyvis, wall_cfg, spawn_heading, gap_y,
                 network, rm, t5a_idx, t5b_idx, fv_state_init, label):
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2, marker_colors=[],
    )
    if use_wall:
        arena.root_element.worldbody.add('geom', **wall_cfg)

    fly = Fly(spawn_pos=SPAWN_POS, spawn_orientation=spawn_heading,
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
    crossing = np.where((x[:-1] < 14) & (x[1:] >= 14))[0]
    min_d    = min(dist_hist) if dist_hist else 999
    max_t5r  = max(t5r_hist) if t5r_hist else 0
    max_bias = max(abs(v) for v in bias_hist) if bias_hist else 0

    print(f"\n  [{label}]")
    status = f"Reached food at step {len(dist_hist)}" if reached else f"Did NOT reach food  min={round(min_d,1)}mm"
    print(f"    {status}")
    if use_wall and len(crossing):
        ci    = crossing[0]
        y_c   = y[ci]
        valid = y_c >= gap_y
        tag   = f"VALID (y>={gap_y:.0f}) OK" if valid else f"BLOCKED ZONE (y<{gap_y:.0f}) FAIL"
        print(f"    x=14 crossed step {ci}: y={y_c:.2f}  -> {tag}")
        print(f"    T5 at crossing: L={t5l_hist[ci]:.3f} R={t5r_hist[ci]:.3f} bias={bias_hist[ci]:.3f}")
    elif use_wall:
        print(f"    Never crossed x=14 in {N_STEPS} steps")
    if use_wall:
        zone = (x > 11) & (x < 14)
        if zone.any():
            print(f"    y range at wall approach (x=11..14): {y[zone].min():.2f} .. {y[zone].max():.2f}")
        print(f"    Max T5_R near wall: {max_t5r:.3f}   Max |bias|: {max_bias:.3f}")

    return reached, (y[crossing[0]] if len(crossing) else None)


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

results = {}

# ── Option A ───────────────────────────────────────────────────────────────────
print("=" * 65)
print("OPTION A: Extended wall (gap at y>=10), heading=0.588 rad")
print("  Wall: x=14, y=0..10, base=-1mm")
print("=" * 65)

ra_a, yc_a = run_scenario(False, False, WALL_A, HEADING_A, GAP_A,
                           network, rm, t5a_idx, t5b_idx, fv_state_init,
                           "A1: no wall, heading=0.588")
rb_a, yc_b = run_scenario(True, True, WALL_A, HEADING_A, GAP_A,
                           network, rm, t5a_idx, t5b_idx, fv_state_init,
                           "A2: wall + flyvis T5")
rc_a, yc_c = run_scenario(True, False, WALL_A, HEADING_A, GAP_A,
                           network, rm, t5a_idx, t5b_idx, fv_state_init,
                           "A3: wall, odor only")
results['A'] = (ra_a, rb_a, rc_a)

# ── Option B ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("OPTION B: Same wall (gap at y>=9), heading=0.3 rad (~17 deg)")
print("  Wall: x=14, y=0..9, base=-1mm  |  Heading more eastward")
print("=" * 65)

ra_b, _ = run_scenario(False, False, WALL_B, HEADING_B, GAP_B,
                        network, rm, t5a_idx, t5b_idx, fv_state_init,
                        "B1: no wall, heading=0.3")
rb_b, _ = run_scenario(True, True, WALL_B, HEADING_B, GAP_B,
                        network, rm, t5a_idx, t5b_idx, fv_state_init,
                        "B2: wall + flyvis T5")
rc_b, _ = run_scenario(True, False, WALL_B, HEADING_B, GAP_B,
                        network, rm, t5a_idx, t5b_idx, fv_state_init,
                        "B3: wall, odor only")
results['B'] = (ra_b, rb_b, rc_b)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
for opt, (r_base, r_flyvis, r_odor) in results.items():
    label_map = {'A': 'Extended wall  (gap y>=10, heading=0.588)',
                 'B': 'Lower heading  (gap y>=9,  heading=0.3) '}
    ok = lambda b: "OK" if b else "FAIL"
    print(f"\n  Option {opt}: {label_map[opt]}")
    print(f"    No wall:        {ok(r_base)}")
    print(f"    Wall + flyvis:  {ok(r_flyvis)}")
    print(f"    Wall odor-only: {ok(r_odor)}")
    all_ok = r_base and r_flyvis and r_odor
    if all_ok:
        print(f"    -> ALL PASS - viable config")
    elif r_base and r_flyvis:
        print(f"    -> Flyvis navigates, odor-only fails - T5 reflex is doing real work")
    elif r_base and not r_flyvis and not r_odor:
        print(f"    -> Both fail - detour too large or gap unreachable")
    else:
        print(f"    -> Mixed - investigate")

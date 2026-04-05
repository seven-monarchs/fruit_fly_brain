"""
tests/test_wall_heading04.py

Tests heading=0.4 rad (~23 deg, intermediate between 0.3 and 0.588) with
the current proven wall config (y=0..9, gap at y>=9, base=-1mm).

Hypothesis:
  - heading=0.3 crashes with BADQACC when flyvis fires (too steep wall contact)
  - heading=0.588 gives only a 2mm detour (barely visible)
  - heading=0.4 should be stable AND give a longer detour (~3mm)

Fly hits wall at lower y than 0.588, needs more northward travel to exit gap,
while wall contact angle is less extreme than 0.3 -> hopefully no BADQACC.

3 conditions:
  A - no wall (baseline)
  B - wall + flyvis T5
  C - wall, odor only

Run:
    wenv310\Scripts\python.exe tests/test_wall_heading04.py
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
FOOD_POS         = np.array([18.0, 12.0, 0.0])
SPAWN_POS        = (0, 0, 0.2)
SPAWN_HEADING    = (0, 0, 0.4)   # ~23 deg, intermediate between 0.3 and 0.588
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

# Proven wall config: y=0..9, base=-1mm
WALL_CFG = {
    'type': 'box', 'pos': [14.0, 4.5, 3.0], 'size': [0.3, 4.5, 4.0],
    'rgba': [0.15, 0.10, 0.05, 1.0], 'contype': 1, 'conaffinity': 1,
    'solimp': '0.9 0.999 0.001 0.5 2', 'solref': '0.02 1',
}
GAP_Y = 9.0

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]


def run_scenario(use_wall, use_flyvis, network, rm, t5a_idx, t5b_idx,
                 fv_state_init, label):
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2, marker_colors=[],
    )
    if use_wall:
        arena.root_element.worldbody.add('geom', **WALL_CFG)

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
        valid = y_c >= GAP_Y
        tag   = f"VALID (y>={GAP_Y:.0f}) OK" if valid else f"BLOCKED ZONE (y<{GAP_Y:.0f}) FAIL"
        print(f"    x=14 crossed step {ci}: y={y_c:.2f}  -> {tag}")
        print(f"    T5 at crossing: L={t5l_hist[ci]:.3f} R={t5r_hist[ci]:.3f} bias={bias_hist[ci]:.3f}")
    elif use_wall:
        print(f"    Never crossed x=14 in {N_STEPS} steps")
    if use_wall:
        zone = (x > 11) & (x < 14)
        if zone.any():
            y_min = y[zone].min()
            y_max = y[zone].max()
            detour = y_max - y_min
            print(f"    y range at wall approach (x=11..14): {y_min:.2f} .. {y_max:.2f}  (span={detour:.2f}mm)")
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

print("=" * 65)
print("Heading=0.4 rad (~23 deg) | Wall: y=0..9, gap at y>=9, base=-1mm")
print("=" * 65)
print(f"Current proven config (heading=0.588) hits wall at y~7, exits y~10 (2mm detour)")
print(f"This test: fly aims more eastward -> hits wall lower -> longer detour\n")

r_a, _   = run_scenario(False, False, network, rm, t5a_idx, t5b_idx, fv_state_init,
                          "A: no wall, heading=0.4")
r_b, yb  = run_scenario(True, True,  network, rm, t5a_idx, t5b_idx, fv_state_init,
                          "B: wall + flyvis T5")
r_c, yc  = run_scenario(True, False, network, rm, t5a_idx, t5b_idx, fv_state_init,
                          "C: wall, odor only")

print("\n" + "=" * 65)
print("VERDICT  (heading=0.4, wall y=0..9, gap y>=9)")
print("=" * 65)
ok = lambda b: "OK" if b else "FAIL"
print(f"  A (no wall):        {ok(r_a)}")
print(f"  B (wall + flyvis):  {ok(r_b)}  crossing y={round(yb,2) if yb else 'n/a'}")
print(f"  C (wall odor-only): {ok(r_c)}  crossing y={round(yc,2) if yc else 'n/a'}")

if r_a and r_b and r_c:
    print("\n  -> ALL PASS - viable config for longer detour")
elif r_a and r_b and not r_c:
    print("\n  -> Flyvis navigates, odor-only fails - T5 reflex carrying the load")
elif r_a and not r_b and r_c:
    print("\n  -> Odor-only works, flyvis fails - T5 oversteer or BADQACC")
elif r_a and not r_b and not r_c:
    print("\n  -> Both B and C fail - detour too large or gap unreachable at this heading")
else:
    print("\n  -> Mixed result - investigate")

print("\nFor comparison: proven config (heading=0.588) -> A=78, B=89, C=92 steps, crossing y~10.2-10.4")

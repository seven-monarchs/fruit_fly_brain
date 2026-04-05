"""
tests/test_flyvis_reflex_integration.py

Integration test: flyvis connectome-constrained visual network as looming reflex.

Replaces the hand-coded LPTC reflex (background subtraction + LOOM_GAIN) with
the full biological pipeline:

    obs["vision"] (2×721)
        -> RetinaMapper (flygym -> flyvis index convention)
        -> flyvis Network (R1-R8 -> L1-L5 -> Mi/Tm -> T4a/T4b/T4c/T4d + T5a-d)
        -> T5a/T5b asymmetry (left eye OFF motion vs right eye OFF motion)
        -> turn bias

T5 neurons are OFF-pathway motion detectors — they fire when a dark edge
expands in their receptive field. A dark obstacle approaching from the left
causes expanding darkening in the left eye -> higher left T5 activity -> steer right.

Conditions:
  A) Solid wall + hand-coded reflex (our current implementation)
  B) Solid wall + flyvis biological reflex (T5 asymmetry)
  C) No wall + flyvis (baseline)

Wall: x=14mm, y=4..20, base at z=1  (same as test_solid_wall_navigation.py)
Solid contact fix: solimp/solref softening

Run:
    wenv310\\Scripts\\python.exe tests/test_flyvis_reflex_integration.py
"""

# NOTE: flygym must be imported before load_dotenv on Windows
from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

import importlib.util, sys
import numpy as np
import torch

# --- Load RetinaMapper without triggering flygym rendering reinit ---
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

# --- Constants ---
FOOD_POS         = np.array([18.0, 12.0, 0.0])
PHYSICS_TIMESTEP = 1e-4
WALK_AMP         = 0.75
ODOR_TURN_K      = 2.5
FEED_DIST        = 1.2
PHYSICS_STEPS    = 250
N_STEPS          = 350
FLYVIS_DT        = 25e-3   # 25ms per decision step

# Hand-coded reflex params (condition A)
LOOM_THRESHOLD = 0.005
LOOM_GAIN      = 6.0
LOOM_DECAY     = 0.6
LOOM_BG_ALPHA  = 0.85

# flyvis reflex params (condition B)
# T5 activity is ~0.5-2.0, odor turn is ~0-2.5 -> gain ~3-5 keeps them comparable
FLYVIS_T5_GAIN  = 4.0     # scale T5 asymmetry to turn units
FLYVIS_DECAY    = 0.5
FLYVIS_BIAS_MAX = 2.0     # clamp so reflex never overwhelms odor
FLYVIS_N_FRAMES = 8       # rolling window fed to flyvis

# Solid wall (same as test_solid_wall_navigation.py extended wall)
WALL = {
    'type': 'box', 'pos': [14.0, 12.0, 5.0], 'size': [0.3, 8.0, 4.0],
    'rgba': [0.15, 0.10, 0.05, 1.0], 'contype': 1, 'conaffinity': 1,
    'solimp': '0.9 0.999 0.001 0.5 2', 'solref': '0.02 1',
}

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]


def run_handcoded(use_wall, label):
    """Condition A: hand-coded background subtraction reflex."""
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2, marker_colors=[],
    )
    if use_wall:
        arena.root_element.worldbody.add('geom', **WALL)

    fly = Fly(spawn_pos=(0,0,0.2), spawn_orientation=(0,0,0.588),
              contact_sensor_placements=csp, enable_olfaction=True,
              enable_adhesion=True, draw_adhesion=False, enable_vision=True)
    sim = HybridTurningController(fly=fly, arena=arena, timestep=PHYSICS_TIMESTEP)
    obs, _ = sim.reset()
    _names = list(sim.physics.named.data.qpos.axes[0].names)
    root_joint = next((n for n in _names if n.endswith('/') and 'dust' not in n), None)

    loom_bg = None; loom_persist = 0.0
    x_hist = []; y_hist = []; loom_hist = []; dist_hist = []
    reached = False

    for step in range(N_STEPS):
        odor      = obs['odor_intensity'][0]
        lr_asym   = (float(odor[1]+odor[3]) - float(odor[0]+odor[2])) / (sum(odor)+1e-9)
        odor_turn = float(np.tanh(lr_asym * 20.0) * ODOR_TURN_K)

        vis = obs.get('vision')
        if vis is not None:
            curr = vis.mean(axis=2)
            if loom_bg is None: loom_bg = curr.copy()
            delta   = loom_bg - curr
            delta_t = np.where(delta > LOOM_THRESHOLD, delta - LOOM_THRESHOLD, 0.0)
            sig_l   = float(delta_t[0].mean())
            sig_r   = float(delta_t[1].mean())
            loom_persist = loom_persist*LOOM_DECAY + (-LOOM_GAIN*(sig_l - sig_r))
            loom_bg = loom_bg*LOOM_BG_ALPHA + curr*(1.0-LOOM_BG_ALPHA)

        turn_bias = odor_turn + loom_persist
        ctrl = np.array([np.clip(WALK_AMP+turn_bias,0.1,1.0),
                         np.clip(WALK_AMP-turn_bias,0.1,1.0)])
        try:
            for _ in range(PHYSICS_STEPS): obs, _, _, _, _ = sim.step(ctrl)
        except Exception as e:
            print(f"  [{label}] CRASH step {step+1}: {e}"); break

        qpos = sim.physics.named.data.qpos[root_joint]
        pos  = np.array([float(qpos[0]), float(qpos[1])])
        dist = float(np.linalg.norm(pos - FOOD_POS[:2]))
        x_hist.append(pos[0]); y_hist.append(pos[1])
        loom_hist.append(loom_persist); dist_hist.append(dist)
        if dist < FEED_DIST: reached=True; break

    min_d = min(dist_hist) if dist_hist else 999
    print(f"  [{label}]  {'reached food at step '+str(len(dist_hist)) if reached else 'did NOT reach food  min='+str(round(min_d,1))+'mm'}  "
          f"max_loom={max(abs(v) for v in loom_hist):.3f}")
    return x_hist, y_hist, loom_hist, dist_hist, reached


def run_flyvis(use_wall, network, rm, label):
    """Condition B/C: flyvis T5 asymmetry as turn bias."""
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[500.0, 0.0]]),
        diffuse_func=lambda x: x**-2, marker_colors=[],
    )
    if use_wall:
        arena.root_element.worldbody.add('geom', **WALL)

    fly = Fly(spawn_pos=(0,0,0.2), spawn_orientation=(0,0,0.588),
              contact_sensor_placements=csp, enable_olfaction=True,
              enable_adhesion=True, draw_adhesion=False, enable_vision=True)
    sim = HybridTurningController(fly=fly, arena=arena, timestep=PHYSICS_TIMESTEP)
    obs, _ = sim.reset()
    _names = list(sim.physics.named.data.qpos.axes[0].names)
    root_joint = next((n for n in _names if n.endswith('/') and 'dust' not in n), None)

    # Rolling frame buffer: (2 eyes, N_FRAMES, 721 ommatidia)
    vis_buffer = []
    flyvis_persist = 0.0

    x_hist = []; y_hist = []; loom_hist = []; dist_hist = []
    t5_l_hist = []; t5_r_hist = []
    reached = False

    for step in range(N_STEPS):
        odor      = obs['odor_intensity'][0]
        lr_asym   = (float(odor[1]+odor[3]) - float(odor[0]+odor[2])) / (sum(odor)+1e-9)
        odor_turn = float(np.tanh(lr_asym * 20.0) * ODOR_TURN_K)

        flyvis_bias = 0.0
        t5_l = t5_r = 0.0
        vis = obs.get('vision')
        if vis is not None:
            grayscale = vis.max(axis=-1).astype(np.float32)  # (2, 721)
            vis_buffer.append(grayscale)
            if len(vis_buffer) > FLYVIS_N_FRAMES:
                vis_buffer.pop(0)

            if len(vis_buffer) >= 3:
                # Build movie tensor: (2 eyes, T frames, 721)
                buf = np.stack(vis_buffer, axis=1)         # (2, T, 721)
                flyvis_buf = np.zeros_like(buf)
                for t in range(buf.shape[1]):
                    flyvis_buf[:, t, :] = rm.flygym_to_flyvis(buf[:, t, :])

                movie = torch.tensor(flyvis_buf, dtype=torch.float32).unsqueeze(2)
                with torch.no_grad():
                    out = network.simulate(movie, dt=FLYVIS_DT, as_layer_activity=True)

                # T5a = OFF rightward motion detector (responds to dark edges expanding right)
                # T5b = OFF leftward motion detector
                # Left eye (sample 0) vs Right eye (sample 1) at last frame
                t5a = out['T5a']  # (2, T, n_columns)
                t5b = out['T5b']
                t5_l = float((t5a[0,-1].abs() + t5b[0,-1].abs()).mean())
                t5_r = float((t5a[1,-1].abs() + t5b[1,-1].abs()).mean())

                # Steer away from more active eye
                # More left T5 -> dark motion on left -> steer right -> negative bias
                t5_asym = t5_l - t5_r
                flyvis_new = -FLYVIS_T5_GAIN * t5_asym
                flyvis_persist = flyvis_persist * FLYVIS_DECAY + flyvis_new * (1-FLYVIS_DECAY)
                flyvis_persist = float(np.clip(flyvis_persist, -FLYVIS_BIAS_MAX, FLYVIS_BIAS_MAX))

        turn_bias = odor_turn + flyvis_persist
        ctrl = np.array([np.clip(WALK_AMP+turn_bias,0.1,1.0),
                         np.clip(WALK_AMP-turn_bias,0.1,1.0)])
        try:
            for _ in range(PHYSICS_STEPS): obs, _, _, _, _ = sim.step(ctrl)
        except Exception as e:
            print(f"  [{label}] CRASH step {step+1}: {e}"); break

        qpos = sim.physics.named.data.qpos[root_joint]
        pos  = np.array([float(qpos[0]), float(qpos[1])])
        dist = float(np.linalg.norm(pos - FOOD_POS[:2]))
        x_hist.append(pos[0]); y_hist.append(pos[1])
        loom_hist.append(flyvis_persist); dist_hist.append(dist)
        t5_l_hist.append(t5_l); t5_r_hist.append(t5_r)
        if dist < FEED_DIST: reached=True; break

    min_d   = min(dist_hist) if dist_hist else 999
    max_t5l = max(t5_l_hist) if t5_l_hist else 0
    max_t5r = max(t5_r_hist) if t5_r_hist else 0
    print(f"  [{label}]  {'reached food at step '+str(len(dist_hist)) if reached else 'did NOT reach food  min='+str(round(min_d,1))+'mm'}  "
          f"max_T5_L={max_t5l:.3f}  max_T5_R={max_t5r:.3f}  max_bias={max(abs(v) for v in loom_hist):.3f}")
    return x_hist, y_hist, loom_hist, dist_hist, t5_l_hist, t5_r_hist, reached


# ---------------------------------------------------------------------------
print("=" * 65)
print("flyvis biological reflex integration test")
print("Wall: x=14mm, y=4..20, solid (solimp/solref)")
print("Food: (18,12)   Fly: (0,0) heading=0.588 rad")
print("=" * 65)

print("\nLoading flyvis pretrained network (Lappalainen et al. 2024)...")
import flyvis
from flyvis import NetworkView
nv      = NetworkView(flyvis.results_dir / 'flow/0000/000')
network = nv.init_network(checkpoint='best')
rm      = _load_retina_mapper()
print("Network loaded. Cell types: R1-R8 -> L1-L5 -> Mi/Tm -> T4/T5 (65 types)\n")

print("-- A: solid wall + hand-coded reflex --")
xa, ya, la, da, ra = run_handcoded(use_wall=True, label="A: hand-coded")

print("\n-- B: solid wall + flyvis T5 reflex --")
xb, yb, lb, db, t5lb, t5rb, rb = run_flyvis(use_wall=True, network=network, rm=rm, label="B: flyvis")

print("\n-- C: no wall + flyvis (baseline) --")
xc, yc, lc, dc, t5lc, t5rc, rc = run_flyvis(use_wall=False, network=network, rm=rm, label="C: no wall")

# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("COMPARISON")
print("=" * 65)
print(f"\n  A (hand-coded, wall):  steps={len(da)}  reached={ra}")
print(f"  B (flyvis, wall):      steps={len(db)}  reached={rb}")
print(f"  C (flyvis, no wall):   steps={len(dc)}  reached={rc}")

if ra and rb:
    print(f"\n  Both navigated around wall.")
    print(f"  Step difference B-A: {len(db)-len(da):+d}  (+ = flyvis slower, - = faster)")
elif rb and not ra:
    print(f"\n  flyvis reflex succeeded where hand-coded failed!")
elif ra and not rb:
    print(f"\n  Hand-coded succeeded but flyvis reflex could not navigate wall.")
else:
    print(f"\n  Neither reached food — wall fully blocks both methods.")

# T5 activation near wall (x > 11mm)
wall_steps_b = [(i, t5lb[i], t5rb[i], lb[i])
                for i in range(len(xb)) if xb[i] > 11.0]
if wall_steps_b:
    avg_l = np.mean([s[1] for s in wall_steps_b])
    avg_r = np.mean([s[2] for s in wall_steps_b])
    asym  = (avg_l - avg_r) / (avg_l + avg_r + 1e-9)
    print(f"\n  T5 near wall (x>11mm):")
    print(f"    Mean T5_L={avg_l:.4f}  T5_R={avg_r:.4f}  asymmetry={asym:+.4f}")
    if asym > 0.05:
        print("    -> Left eye detects wall -> correct rightward steering")
    elif asym < -0.05:
        print("    -> Right eye detects wall -> leftward steering (wall on right)")
    else:
        print("    -> Symmetric signal (wall approached head-on)")

# Path comparison vs hand-coded
print(f"\n-- Path near wall: A (hand-coded) vs B (flyvis) --")
print(f"  {'step':>4}  {'x_A':>6} {'y_A':>6}  {'x_B':>6} {'y_B':>6}  {'dy':>6}  "
      f"{'T5L':>7} {'T5R':>7}  {'bias_B':>7}")
n = max(len(xa), len(xb))
xa_p = xa + [None]*(n-len(xa)); ya_p = ya + [None]*(n-len(ya))
xb_p = xb + [None]*(n-len(xb)); yb_p = yb + [None]*(n-len(yb))
lb_p = lb + [None]*(n-len(lb))
t5l_p = t5lb + [None]*(n-len(t5lb)); t5r_p = t5rb + [None]*(n-len(t5rb))
printed = 0
for i in range(n):
    in_zone = (xa_p[i] is not None and xa_p[i] > 11) or \
              (xb_p[i] is not None and xb_p[i] > 11)
    if not in_zone: continue
    fmt = lambda v, w=6, d=2: f"{v:{w}.{d}f}" if v is not None else ' '*w
    dy = (yb_p[i] - ya_p[i]) if (yb_p[i] is not None and ya_p[i] is not None) else None
    print(f"  {i+1:>4}  {fmt(xa_p[i])} {fmt(ya_p[i])}  "
          f"{fmt(xb_p[i])} {fmt(yb_p[i])}  "
          f"{fmt(dy, 6, 2)}  "
          f"{fmt(t5l_p[i], 7, 4)} {fmt(t5r_p[i], 7, 4)}  "
          f"{fmt(lb_p[i], 7, 3)}")
    printed += 1
    if printed >= 50: print("  ..."); break

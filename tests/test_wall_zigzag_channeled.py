"""
tests/test_wall_zigzag_channeled.py

Zigzag tunnel layout with wall-channeled odor.

Instead of the odor diffusing through walls (straight-line inverse-square),
this test pre-computes the shortest WALKABLE path distance from food to every
grid cell using Dijkstra.  Walls block the path, so the odor field reflects
the physical corridor:

  - West of wall 1: fly smells odor coming from the gap direction (y~10),
    not directly through the wall toward food at (20,2).
  - Inside tunnel: strong odor pulling south toward wall 2 gap.
  - East of wall 2: direct odor from food.

Layout:
  Wall 1: x=8,  y=0..10  gap at y>=10
  Wall 2: x=14, y=6..16  gap at y<=6
  Tunnel: x=8..14, y=6..10
  Food:   (20, 2)
  Spawn:  (0, 0), heading=0.588

Run:
    wenv310\\Scripts\\python.exe tests/test_wall_zigzag_channeled.py
"""

from flygym import Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

import heapq
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


# ── Grid-based channeled odor ──────────────────────────────────────────────────

GRID_RES  = 0.5          # mm per cell
X_MIN, X_MAX = -1.0, 22.0
Y_MIN, Y_MAX = -16.0, 18.0

# Walls as (x_center, y_center, x_half, y_half) in mm
# Wall 1 extends to y=-15 so south route is 52mm vs north gap 28mm -> fly goes north
# Wall 2: x=14, y=6..16 -> gap at y<=6
WALL_RECTS = [
    (8.0,  -2.5, 0.4, 12.5),  # spans y=-15..10, gap at y>=10
    (14.0, 11.0, 0.4,  5.0),  # spans y=6..16,   gap at y<=6
]


def build_odor_field(food_xy, wall_rects, grid_res=GRID_RES,
                     x_range=(X_MIN, X_MAX), y_range=(Y_MIN, Y_MAX),
                     peak=500.0):
    """
    Dijkstra from food through walkable cells.
    Returns (odor_field, xs, ys) where odor_field[ix,iy] = peak/dist^2.
    """
    xs = np.arange(x_range[0], x_range[1] + grid_res * 0.5, grid_res)
    ys = np.arange(y_range[0], y_range[1] + grid_res * 0.5, grid_res)
    NX, NY = len(xs), len(ys)

    # Build wall mask
    blocked = np.zeros((NX, NY), dtype=bool)
    for (wx, wy, whx, why) in wall_rects:
        ix_lo = int(np.floor((wx - whx - x_range[0]) / grid_res))
        ix_hi = int(np.ceil( (wx + whx - x_range[0]) / grid_res))
        iy_lo = int(np.floor((wy - why - y_range[0]) / grid_res))
        iy_hi = int(np.ceil( (wy + why - y_range[0]) / grid_res))
        ix_lo = max(0, ix_lo); ix_hi = min(NX-1, ix_hi)
        iy_lo = max(0, iy_lo); iy_hi = min(NY-1, iy_hi)
        blocked[ix_lo:ix_hi+1, iy_lo:iy_hi+1] = True

    # Food cell
    fx = int(round((food_xy[0] - x_range[0]) / grid_res))
    fy = int(round((food_xy[1] - y_range[0]) / grid_res))
    fx = int(np.clip(fx, 0, NX-1))
    fy = int(np.clip(fy, 0, NY-1))

    # Dijkstra
    INF = 1e9
    dist = np.full((NX, NY), INF)
    dist[fx, fy] = 0.0
    pq = [(0.0, fx, fy)]
    moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
             (-1,-1,1.414),(-1,1,1.414),(1,-1,1.414),(1,1,1.414)]

    while pq:
        d, ix, iy = heapq.heappop(pq)
        if d > dist[ix, iy] + 1e-9:
            continue
        for ddx, ddy, cost in moves:
            nx2 = ix + ddx
            ny2 = iy + ddy
            if 0 <= nx2 < NX and 0 <= ny2 < NY and not blocked[nx2, ny2]:
                nd = d + cost * grid_res
                if nd < dist[nx2, ny2]:
                    dist[nx2, ny2] = nd
                    heapq.heappush(pq, (nd, nx2, ny2))

    # Odor intensity: peak / dist^2, clamped at minimum distance 1mm
    safe_dist = np.maximum(dist, 1.0)
    odor_field = np.where(dist < INF, peak / safe_dist**2, 0.0)
    return odor_field, xs, ys, blocked


def lookup_odor(px, py, odor_field, xs, ys):
    """Bilinear interpolation of odor_field at world position (px, py)."""
    px = float(np.clip(px, xs[0], xs[-1]))
    py = float(np.clip(py, ys[0], ys[-1]))
    res_x = xs[1] - xs[0]
    res_y = ys[1] - ys[0]
    ix = (px - xs[0]) / res_x
    iy = (py - ys[0]) / res_y
    ix0 = int(ix);  iy0 = int(iy)
    ix1 = min(ix0+1, len(xs)-1)
    iy1 = min(iy0+1, len(ys)-1)
    fx = ix - ix0;  fy = iy - iy0
    return (odor_field[ix0,iy0]*(1-fx)*(1-fy) +
            odor_field[ix1,iy0]*fx*(1-fy) +
            odor_field[ix0,iy1]*(1-fx)*fy +
            odor_field[ix1,iy1]*fx*fy)


def print_odor_map(odor_field, xs, ys, blocked, food_xy, spawn_xy):
    """Print ASCII map of the odor field for visual verification."""
    print("\nOdor field (X axis right, Y axis up):")
    print("  High odor = # , Medium = + , Low = . , Wall = | , Zero = ' '")
    # Sample every 2mm for readability
    step = max(1, int(2.0 / (xs[1]-xs[0])))
    xi_range = range(0, len(xs), step)
    yi_range = range(len(ys)-1, -1, -step)

    for iy in yi_range:
        y = ys[iy]
        row = f"{y:4.0f}|"
        for ix in xi_range:
            x = xs[ix]
            if blocked[ix, iy]:
                row += '#'
            else:
                v = odor_field[ix, iy]
                # mark food and spawn
                if abs(x - food_xy[0]) < 1.5 and abs(y - food_xy[1]) < 1.5:
                    row += 'F'
                elif abs(x - spawn_xy[0]) < 1.5 and abs(y - spawn_xy[1]) < 1.5:
                    row += 'S'
                elif v > 50:
                    row += '@'
                elif v > 10:
                    row += '#'
                elif v > 3:
                    row += '+'
                elif v > 0.5:
                    row += '.'
                else:
                    row += ' '
        print(row)
    print("    +" + "-" * len(xi_range))
    x_labels = "     "
    for ix in xi_range:
        x_labels += str(int(round(xs[ix])))[-1]
    print(x_labels)
    print()


# ── Simulation constants ───────────────────────────────────────────────────────
FOOD_POS         = np.array([20.0, 2.0, 0.0])
SPAWN_POS        = (0, 0, 0.2)
SPAWN_HEADING    = (0, 0, 0.588)
PHYSICS_TIMESTEP = 1e-4
WALK_AMP         = 0.75
ODOR_TURN_K      = 2.5
FEED_DIST        = 1.2
PHYSICS_STEPS    = 250
N_STEPS          = 500

FLYVIS_T5_GAIN  = 0.5
FLYVIS_DECAY    = 0.5
FLYVIS_BIAS_MAX = 0.15
FLYVIS_DT       = 25e-3
ANT_SEP         = 0.5    # mm, lateral antenna separation

WALL_1 = {
    'name': 'wall1',
    'type': 'box', 'pos': [8.0, -2.5, 3.0], 'size': [0.3, 12.5, 4.0],  # y=-15..10
    'rgba': [0.15, 0.10, 0.05, 1.0], 'contype': 1, 'conaffinity': 1,
    'solimp': '0.9 0.999 0.001 0.5 2', 'solref': '0.02 1',
}
WALL_2 = {
    'name': 'wall2',
    'type': 'box', 'pos': [14.0, 11.0, 3.0], 'size': [0.3, 5.0, 4.0],
    'rgba': [0.15, 0.10, 0.05, 1.0], 'contype': 1, 'conaffinity': 1,
    'solimp': '0.9 0.999 0.001 0.5 2', 'solref': '0.02 1',
}

csp = [f'{l}{s}' for l in ['LF','LM','LH','RF','RM','RH']
       for s in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]


def run_scenario(use_walls, use_flyvis, odor_field, xs, ys,
                 network, rm, t5a_idx, t5b_idx, fv_state_init, label):
    arena = OdorArena(
        odor_source=FOOD_POS[np.newaxis],
        peak_odor_intensity=np.array([[1.0, 0.0]]),   # dummy - not used for steering
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
        # ── Channeled odor steering (replaces obs['odor_intensity']) ──
        qpos    = sim.physics.named.data.qpos[root_joint]
        pos_x   = float(qpos[0])
        pos_y   = float(qpos[1])
        qw      = float(qpos[3])
        qz      = float(qpos[6])
        heading = float(np.arctan2(2*qw*qz, 1 - 2*qz**2))

        # lateral unit vector (right of fly)
        right_x =  np.sin(heading)
        right_y = -np.cos(heading)

        odor_l = lookup_odor(pos_x - ANT_SEP*right_x,
                             pos_y - ANT_SEP*right_y, odor_field, xs, ys)
        odor_r = lookup_odor(pos_x + ANT_SEP*right_x,
                             pos_y + ANT_SEP*right_y, odor_field, xs, ys)

        total  = odor_l + odor_r + 1e-9
        lr_asym   = (odor_r - odor_l) / total
        odor_turn = float(np.tanh(lr_asym * 20.0) * ODOR_TURN_K)

        # ── flyvis T5 ──────────────────────────────────────────────────
        t5_l = t5_r = 0.0
        vis = obs.get('vision')
        if vis is not None and use_flyvis:
            vis_gray   = vis.max(axis=2).astype(np.float32)
            vis_mapped = rm.flygym_to_flyvis(vis_gray)
            frame_t    = torch.tensor(vis_mapped, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
            with torch.no_grad():
                network.stimulus.zero(2, 1)
                network.stimulus.add_input(frame_t)
                fv_states = network.forward(network.stimulus(), FLYVIS_DT,
                                            state=fv_state, as_states=True)
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

        qpos2 = sim.physics.named.data.qpos[root_joint]
        pos   = np.array([float(qpos2[0]), float(qpos2[1])])
        dist  = float(np.linalg.norm(pos - FOOD_POS[:2]))
        x_hist.append(pos[0]); y_hist.append(pos[1])
        dist_hist.append(dist)
        t5l_hist.append(t5_l); t5r_hist.append(t5_r)
        bias_hist.append(loom_persist)
        if dist < FEED_DIST:
            reached = True
            break

    x = np.array(x_hist); y = np.array(y_hist)
    cross1 = np.where((x[:-1] < 8)  & (x[1:] >= 8))[0]
    cross2 = np.where((x[:-1] < 14) & (x[1:] >= 14))[0]
    min_d  = min(dist_hist) if dist_hist else 999
    max_y  = max(y_hist) if y_hist else 0

    print(f"\n  [{label}]")
    status = (f"Reached food at step {len(dist_hist)}" if reached
              else f"Did NOT reach food  min={round(min_d,1)}mm")
    print(f"    {status}")
    print(f"    Max y reached: {max_y:.2f}mm")

    if use_walls:
        if len(cross1):
            ci   = cross1[0]
            y_c1 = y[ci]
            tag  = "VALID (y>=10) OK" if y_c1 >= 10 else f"WRONG (y<10={y_c1:.2f}) FAIL"
            print(f"    Wall 1 crossed step {ci}: y={y_c1:.2f}  -> {tag}")
        else:
            print(f"    Wall 1 (x=8): never crossed")

        if len(cross2):
            ci   = cross2[0]
            y_c2 = y[ci]
            tag  = "VALID (y<=6) OK" if y_c2 <= 6 else f"WRONG (y>6={y_c2:.2f}) FAIL"
            print(f"    Wall 2 crossed step {ci}: y={y_c2:.2f}  -> {tag}")
        else:
            print(f"    Wall 2 (x=14): never crossed")

        tunnel = (x >= 8) & (x <= 14)
        if tunnel.any():
            print(f"    y in tunnel (x=8..14): {y[tunnel].min():.2f} .. {y[tunnel].max():.2f}")

    return reached


# ── Pre-compute odor field ─────────────────────────────────────────────────────
print("Building channeled odor field (Dijkstra)...")
odor_field, xs, ys, blocked = build_odor_field(
    food_xy=FOOD_POS[:2],
    wall_rects=WALL_RECTS,
)
print_odor_map(odor_field, xs, ys, blocked,
               food_xy=FOOD_POS[:2], spawn_xy=np.array([0.0, 0.0]))

# Spot-check: odor at spawn vs odor at wall 1 gap
o_spawn = lookup_odor(0.0, 0.0, odor_field, xs, ys)
o_gap1  = lookup_odor(7.0, 10.0, odor_field, xs, ys)   # just west of wall 1 gap
o_tunnel= lookup_odor(11.0, 8.0, odor_field, xs, ys)   # inside tunnel
o_gap2  = lookup_odor(15.0, 5.0, odor_field, xs, ys)   # just east of wall 2 gap
print(f"Odor spot-checks:")
print(f"  spawn  (0,  0): {o_spawn:.3f}")
print(f"  gap1   (7, 10): {o_gap1:.3f}  <- should be > spawn (odor exits here)")
print(f"  tunnel (11, 8): {o_tunnel:.3f}")
print(f"  gap2  (15,  5): {o_gap2:.3f}  <- should be high (near food side)")

# ── Load flyvis ────────────────────────────────────────────────────────────────
print("\nLoading flyvis network...")
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
        fv_state_init = network.forward(network.stimulus(), FLYVIS_DT,
                                        state=fv_state_init, as_states=True)[-1]
print("Ready.\n")

print("=" * 65)
print("ZIGZAG CHANNELED ODOR TEST")
print("  Wall 1: x=8,  y=0..10  gap at y>=10")
print("  Wall 2: x=14, y=6..16  gap at y<=6")
print("  Food:   (20, 2)  |  Odor: wall-channeled (Dijkstra)")
print("=" * 65)

r_a = run_scenario(False, False, odor_field, xs, ys,
                   network, rm, t5a_idx, t5b_idx, fv_state_init,
                   "A: no walls")
r_b = run_scenario(True,  True,  odor_field, xs, ys,
                   network, rm, t5a_idx, t5b_idx, fv_state_init,
                   "B: both walls + flyvis T5")
r_c = run_scenario(True,  False, odor_field, xs, ys,
                   network, rm, t5a_idx, t5b_idx, fv_state_init,
                   "C: both walls, odor only")

print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
ok = lambda b: "OK" if b else "FAIL"
print(f"  A (no walls):        {ok(r_a)}")
print(f"  B (walls + flyvis):  {ok(r_b)}")
print(f"  C (walls odor-only): {ok(r_c)}")

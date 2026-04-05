"""
V4 — Closed-loop olfactory-driven brain-body simulation.

Brain and body share a SINGLE continuous timeline (no tiling, no looping).
Brian2 runs for exactly the same duration as the physics simulation.

Brain panel layers (color-coded by circuit):
  ● grey/cyan  — all neurons, base LIF spike activity
  ● green      — descending neurons (DN) — locomotion control
  ● blue       — olfactory neurons (ORN/PN) — boosted when fly detects odor
  ● orange     — SEZ / feeding neurons — boosted during feeding phase

Output: split-screen video (brain LEFT | fly isometric view RIGHT).
"""

import sys, re, time, importlib.util, heapq
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

# ── Tee stdout + stderr → console AND timestamped log file ───────────────────
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_PATH = _LOG_DIR / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_run.log"

class _Tee:
    """Mirror every write to both the original stream and a log file."""
    def __init__(self, original, path, mode="w"):
        self._orig = original
        self._file = open(path, mode, buffering=1, encoding="utf-8")
    def write(self, msg):
        self._orig.write(msg)
        self._file.write(msg)
    def flush(self):
        self._orig.flush()
        self._file.flush()
    def fileno(self):
        return self._orig.fileno()   # lets C extensions get the real fd

sys.stdout = _Tee(sys.__stdout__, _LOG_PATH, mode="w")
sys.stderr = _Tee(sys.__stderr__, _LOG_PATH, mode="a")   # append to same file
print(f"[log] {_LOG_PATH}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio

# ── flygym must come before load_dotenv (Windows GL context) ──────────────────
import flygym.preprogrammed as preprogrammed
from flygym import Fly, YawOnlyCamera
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController
from dotenv import load_dotenv
load_dotenv()

# ── Brian2 backend selection ──────────────────────────────────────────────────
# run.bat sets BRIAN2_NUMPY_FALLBACK=1 when no C++ compiler is detected.
# You can also force numpy manually by uncommenting the line below.
import os as _os
if _os.environ.get("BRIAN2_NUMPY_FALLBACK") == "1":
    from brian2 import prefs as _b2prefs; _b2prefs.codegen.target = "numpy"
    print("[Brian2] numpy backend (no C++ compiler found)")
else:
    print("[Brian2] C++ backend (default)")
# To force numpy manually: from brian2 import prefs; prefs.codegen.target = "numpy"
from brian2 import (
    NeuronGroup, Synapses, PoissonInput, PoissonGroup, SpikeMonitor, Network,
    mV, ms, Hz, second, start_scope,
)

BRAIN_DIR = Path(__file__).parent / "brain_model"
PATH_COMP  = BRAIN_DIR / "Completeness_783.csv"
PATH_CON   = BRAIN_DIR / "Connectivity_783.parquet"
sys.path.insert(0, str(BRAIN_DIR))
from model import create_model, poi, default_params

# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
PHYSICS_TIMESTEP           = 1e-4        # s
DECISION_INTERVAL          = 0.025       # s — one CPG step
PHYSICS_STEPS_PER_DECISION = int(DECISION_INTERVAL / PHYSICS_TIMESTEP)   # 250
PLAY_SPEED                 = 0.25        # video slowed 4× vs real time
FPS                        = 30
PHYS_DURATION_S            = 10.0        # physics simulation length (s)
N_DECISIONS_TOTAL          = int(PHYS_DURATION_S / DECISION_INTERVAL)    # 400

# Brain runs for EXACTLY the same duration as physics (no tiling/repeating)
BRAIN_DURATION_S           = PHYS_DURATION_S
BRAIN_DURATION_MS          = int(BRAIN_DURATION_S * 1000)                # 10 000 ms
STIM_RATE_HZ               = 150
N_BRAIN_DECISIONS          = int(BRAIN_DURATION_S / DECISION_INTERVAL)   # 400 (same as body)

WALK_AMP     = 0.75
ODOR_TURN_K  = 2.5
FOOD_POS     = np.array([20.0, 2.0, 0.0])
FEED_DIST    = 1.2     # mm — proximity trigger
FEED_DUR     = 2.0     # s

FPS_V          = FPS
FRAME_DT_MS    = 1000.0 / FPS_V                                          # ≈33.3 ms
DECAY_TAU_MS   = 80      # glow decay

# Visual dot sizes
DOT_BG_SIZE    = 1.0     # background (all neurons, always visible)
DOT_LIF_SIZE   = 14.0    # LIF spike glow (white/cyan layer)
DOT_DN_SIZE    = 20.0    # descending-neuron highlight (green)
DOT_OLF_SIZE   = 20.0    # olfactory highlight (blue)
DOT_SEZ_SIZE   = 20.0    # SEZ/feeding highlight (orange)
DOT_VIS_SIZE   = 20.0    # visual/lamina highlight (yellow)

VIS_STIM_MIN_HZ = 20.0   # lamina firing rate at zero luminance
VIS_STIM_MAX_HZ = 150.0  # lamina firing rate at full luminance (float32 1.0)

# Visual obstacle avoidance — flyvis connectome-constrained T5 motion detector
# Lappalainen et al. 2024 (Nature): R1-R8 -> L1-L5 -> Mi/Tm -> T4/T5 (65 cell types)
# T5 = OFF-pathway motion detector; asymmetry between eyes drives avoidance turn
FLYVIS_T5_GAIN  = 0.5    # scale T5 asymmetry to turn units (gentle — prevents oversteer near wall)
FLYVIS_DECAY    = 0.5    # exponential decay of persisted bias per step
FLYVIS_BIAS_MAX = 0.15   # clamp so reflex assists odor without dominating
FLYVIS_DT       = 25e-3  # integration dt (one frame = one 25ms decision step)


def _load_retina_mapper():
    """Load flygym RetinaMapper via importlib to avoid MuJoCo rendering reinit."""
    spec = importlib.util.spec_from_file_location(
        'flygym.vision.retina',
        Path(__file__).parent / 'wenv310/lib/site-packages/flygym/vision/retina.py')
    retina_mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('flygym.vision.retina', retina_mod)
    sys.modules.setdefault('flygym.vision', retina_mod)
    spec.loader.exec_module(retina_mod)
    spec2 = importlib.util.spec_from_file_location(
        'vision_network',
        Path(__file__).parent / 'wenv310/lib/site-packages/flygym/examples/vision/vision_network.py')
    vn_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(vn_mod)
    return vn_mod.RetinaMapper()

# ── Channeled odor grid (Dijkstra path-distance field) ───────────────────────
# Walls block straight-line diffusion; odor routes through physical corridor.
# Wall 1: x=8,  y=-15..10  gap at y>=10  (north)
# Wall 2: x=14, y=6..16    gap at y<=6   (south into tunnel)
GRID_RES         = 0.5          # mm per cell
X_MIN, X_MAX     = -1.0, 22.0
Y_MIN, Y_MAX     = -16.0, 18.0
ANT_SEP          = 0.5          # mm, lateral antenna separation

# Walls as (x_center, y_center, x_half, y_half) in mm for odor grid
_ODOR_WALL_RECTS = [
    (8.0,  -2.5, 0.4, 12.5),  # Wall 1: y=-15..10, gap y>=10
    (14.0, 11.0, 0.4,  5.0),  # Wall 2: y=6..16,   gap y<=6
]


def build_odor_field(food_xy, wall_rects=_ODOR_WALL_RECTS, grid_res=GRID_RES,
                     x_range=(X_MIN, X_MAX), y_range=(Y_MIN, Y_MAX), peak=500.0):
    """Dijkstra shortest walkable path from food. Returns (odor_field, xs, ys, blocked)."""
    xs = np.arange(x_range[0], x_range[1] + grid_res * 0.5, grid_res)
    ys = np.arange(y_range[0], y_range[1] + grid_res * 0.5, grid_res)
    NX, NY = len(xs), len(ys)

    blocked = np.zeros((NX, NY), dtype=bool)
    for (wx, wy, whx, why) in wall_rects:
        ix_lo = max(0, int(np.floor((wx - whx - x_range[0]) / grid_res)))
        ix_hi = min(NX-1, int(np.ceil( (wx + whx - x_range[0]) / grid_res)))
        iy_lo = max(0, int(np.floor((wy - why - y_range[0]) / grid_res)))
        iy_hi = min(NY-1, int(np.ceil( (wy + why - y_range[0]) / grid_res)))
        blocked[ix_lo:ix_hi+1, iy_lo:iy_hi+1] = True

    fx = int(np.clip(round((food_xy[0] - x_range[0]) / grid_res), 0, NX-1))
    fy = int(np.clip(round((food_xy[1] - y_range[0]) / grid_res), 0, NY-1))

    INF  = 1e9
    dist = np.full((NX, NY), INF)
    dist[fx, fy] = 0.0
    pq   = [(0.0, fx, fy)]
    moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
             (-1,-1,1.414),(-1,1,1.414),(1,-1,1.414),(1,1,1.414)]
    while pq:
        d, ix, iy = heapq.heappop(pq)
        if d > dist[ix, iy] + 1e-9:
            continue
        for ddx, ddy, cost in moves:
            nx2, ny2 = ix + ddx, iy + ddy
            if 0 <= nx2 < NX and 0 <= ny2 < NY and not blocked[nx2, ny2]:
                nd = d + cost * grid_res
                if nd < dist[nx2, ny2]:
                    dist[nx2, ny2] = nd
                    heapq.heappush(pq, (nd, nx2, ny2))

    safe_dist  = np.maximum(dist, 1.0)
    odor_field = np.where(dist < INF, peak / safe_dist**2, 0.0)
    return odor_field, xs, ys, blocked


def lookup_odor(px, py, odor_field, xs, ys):
    """Bilinear interpolation of odor_field at world position (px, py)."""
    px   = float(np.clip(px, xs[0], xs[-1]))
    py   = float(np.clip(py, ys[0], ys[-1]))
    rx   = xs[1] - xs[0];  ry = ys[1] - ys[0]
    ix   = (px - xs[0]) / rx;  iy = (py - ys[0]) / ry
    ix0  = int(ix);  iy0 = int(iy)
    ix1  = min(ix0+1, len(xs)-1);  iy1 = min(iy0+1, len(ys)-1)
    fx   = ix - ix0;  fy = iy - iy0
    return (odor_field[ix0, iy0]*(1-fx)*(1-fy) +
            odor_field[ix1, iy0]*fx*(1-fy) +
            odor_field[ix0, iy1]*(1-fx)*fy +
            odor_field[ix1, iy1]*fx*fy)


BRAIN_PANEL_W  = 1280   # spans full output width (both fly panels combined)
BRAIN_PANEL_H  = 480
MIN_AMP        = 0.3

# ═══════════════════════════════════════════════════════════════════════════════
#  1. CONNECTOME + ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════════════════
T_START = time.time()

def _elapsed(since):
    s = time.time() - since
    return f"{int(s//60)}m {s%60:.0f}s"

print("Loading connectome ...")
df_comp   = pd.read_csv(PATH_COMP, index_col=0)
n_neurons = len(df_comp)
flyid2i   = {flyid: i for i, flyid in enumerate(df_comp.index)}
print(f"  {n_neurons:,} neurons")

print("Loading annotations ...")
df_ann = pd.read_csv(BRAIN_DIR / "flywire_annotations.tsv", sep="\t", low_memory=False)
df_ann["brian_idx"] = df_ann["root_id"].map(flyid2i)

# ── neuron groups (all extracted BEFORE soma-coord filter) ────────────────────
# Ascending sensory input — Poisson stimulation
ascending_ids = df_ann[df_ann["super_class"] == "ascending"]["root_id"].tolist()
sensory_idx   = [flyid2i[fid] for fid in ascending_ids if fid in flyid2i]
print(f"  Ascending (stimulation input): {len(sensory_idx):,}")

def _find_ids(pattern, cols=("cell_class", "cell_type", "super_class")):
    """Return brian indices matching regex pattern in any annotation column."""
    mask = pd.Series(False, index=df_ann.index)
    for col in cols:
        if col in df_ann.columns:
            mask |= df_ann[col].astype(str).str.lower().str.contains(
                pattern, na=False, regex=True
            )
    return df_ann[mask]["brian_idx"].dropna().astype(int).tolist()

# Olfactory — ORNs, uniglomerular/multiglomerular PNs
olfactory_brian = _find_ids(r"olfactory|olfactori|\born\b|projection.neuron")
print(f"  Olfactory neurons: {len(olfactory_brian):,}")

# SEZ / feeding — subesophageal zone, pharyngeal sensory, Fdg, gustatory
sez_brian = _find_ids(r"sez|fdg|feeding|subesophageal|pharyngeal|gustatory")
print(f"  SEZ/feeding neurons: {len(sez_brian):,}")

# Descending neurons — motor readout
dn_left_ids, dn_right_ids, dn_both_ids = [], [], []
dn_csv = BRAIN_DIR / "descending_neurons.csv"
if dn_csv.exists():
    dn_df = pd.read_csv(dn_csv)
    for _, row in dn_df.iterrows():
        fid = row["root_id"]
        if fid not in flyid2i:
            continue
        bidx = flyid2i[fid]
        side = str(row.get("side", "")).strip().lower()
        if   side == "left":  dn_left_ids.append(bidx)
        elif side == "right": dn_right_ids.append(bidx)
        else:                 dn_both_ids.append(bidx)
dn_all_ids = dn_left_ids + dn_right_ids + dn_both_ids
print(f"  DNs: {len(dn_left_ids)} L / {len(dn_right_ids)} R / {len(dn_both_ids)} bilateral")

# Visual input — lamina→medulla (LA>ME) neurons: first connectome synapse of photoreceptors.
# Split left/right so each eye drives its ipsilateral optic lobe.
_lam = df_ann[df_ann["cell_class"] == "LA>ME"].copy()
_lam = _lam[_lam["brian_idx"].notna()]
_lam["brian_idx"] = _lam["brian_idx"].astype(int)
_lam_side = _lam["side"].astype(str).str.strip().str.lower()
lam_left_ids  = _lam[_lam_side == "left"] ["brian_idx"].tolist()
lam_right_ids = _lam[_lam_side == "right"]["brian_idx"].tolist()
lam_both_ids  = _lam[~_lam_side.isin(["left", "right"])]["brian_idx"].tolist()
lam_all_ids   = lam_left_ids + lam_right_ids + lam_both_ids
print(f"  LA>ME lamina neurons: {len(lam_left_ids)} L / {len(lam_right_ids)} R / {len(lam_both_ids)} bilateral")

# ═══════════════════════════════════════════════════════════════════════════════
#  2. SOMA POSITIONS (frontal view)
# ═══════════════════════════════════════════════════════════════════════════════
# Use soma_x/soma_y where available; fall back to pos_x/pos_y for the rest.
# soma coords are in 4 nm/vox space; pos_x/pos_y are in 4 nm/vox too (same space).
df_work = df_ann[df_ann["brian_idx"].notna()].copy()
df_work["brian_idx"] = df_work["brian_idx"].astype(int)
# Merged x: soma_x first, then pos_x
df_work["mx"] = df_work["soma_x"].where(df_work["soma_x"].notna(), df_work["pos_x"])
df_work["my"] = df_work["soma_y"].where(df_work["soma_y"].notna(), df_work["pos_y"])
df_pos = df_work.dropna(subset=["mx", "my"])
pos_x = np.full(n_neurons, np.nan)
pos_y = np.full(n_neurons, np.nan)
pos_x[df_pos["brian_idx"].values] = df_pos["mx"].values
pos_y[df_pos["brian_idx"].values] = df_pos["my"].values
valid_mask = ~np.isnan(pos_x)
valid_idx  = np.where(valid_mask)[0]
dense_map  = {int(b): i for i, b in enumerate(valid_idx)}
px = pos_x[valid_idx]
pz = -pos_y[valid_idx]   # flip so dorsal = up
n_valid = len(valid_idx)
print(f"  {n_valid:,} neurons with positions (soma or centroid fallback)")

def _to_dense(brian_ids):
    arr = np.array([dense_map[b] for b in brian_ids if b in dense_map], dtype=int)
    return arr

olfactory_dense = _to_dense(olfactory_brian)
sez_dense       = _to_dense(sez_brian)
dn_dense        = _to_dense(dn_all_ids)
lam_dense       = _to_dense(lam_all_ids)
print(f"  ● Olfactory (positioned): {len(olfactory_dense):,}")
print(f"  ● SEZ/feeding (positioned): {len(sez_dense):,}")
print(f"  ● DN (positioned): {len(dn_dense):,}")
print(f"  ● LA>ME lamina/visual (positioned): {len(lam_dense):,}")

# ═══════════════════════════════════════════════════════════════════════════════
#  3. BRIAN2 — create network (will run interleaved with physics, 25 ms/step)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Brian2 LIF network ...")
_t_brian = time.time()
start_scope()

SENSORY_STIM_RATE  = STIM_RATE_HZ  # Hz — ascending neurons (dynamic via proprio)
CIRCUIT_STIM_RATE  = 80            # Hz — olfactory + SEZ (fixed baseline)
PROPRIO_MIN        = 0.15          # min fraction of base rate during stillness
MAX_JOINT_VELOCITY = 8.0           # rad/s — reference for velocity normalisation

circuit_stim_idx = list(set(olfactory_brian + sez_brian))
print(f"  Circuit neurons @ {CIRCUIT_STIM_RATE} Hz: {len(circuit_stim_idx):,}")
params = {
    **default_params,
    "t_run" : BRAIN_DURATION_MS * ms,
    "n_run" : 1,
    "r_poi" : SENSORY_STIM_RATE * Hz,
    "r_poi2": CIRCUIT_STIM_RATE * Hz,
}
neu, syn, spk_mon = create_model(PATH_COMP, PATH_CON, params)

# ── Ascending neurons: PoissonGroup with proprioceptive feedback ─────────────
# Rates are updated each step via the property setter (asc_group.rates = x * Hz).
# Tested in tests/test_poisson_rate_update.py — this does NOT trigger Brian2
# recompilation; Brian2 updates the underlying variable buffer in-place.
n_asc     = len(sensory_idx)
asc_group = PoissonGroup(N=n_asc, rates=SENSORY_STIM_RATE * Hz)
asc_syn   = Synapses(asc_group, neu, 'w : volt', on_pre='v += w',
                     delay=params['t_dly'])
asc_syn.connect(i=np.arange(n_asc), j=np.array(sensory_idx))
asc_syn.w = params['w_syn'] * params['f_poi']
for bidx in sensory_idx:          # no refractory period for Poisson targets
    neu[bidx].rfc = 0 * ms

# ── Circuit neurons (olfactory + SEZ): fixed rate via poi()
poi_circuit, neu = poi(neu, [], circuit_stim_idx, params)  # exc=[] — only exc2

# ── Visual neurons (LA>ME lamina): PoissonGroup driven by compound-eye luminance
# Left eye  → lam_left_ids  (ipsilateral optic lobe)
# Right eye → lam_right_ids (ipsilateral optic lobe)
# Bilateral / unassigned neurons receive average of both eyes.
n_lam_l = len(lam_left_ids)
n_lam_r = len(lam_right_ids)
n_lam_b = len(lam_both_ids)

vis_group_l = PoissonGroup(N=max(n_lam_l, 1), rates=VIS_STIM_MIN_HZ * Hz)
vis_group_r = PoissonGroup(N=max(n_lam_r, 1), rates=VIS_STIM_MIN_HZ * Hz)
vis_group_b = PoissonGroup(N=max(n_lam_b, 1), rates=VIS_STIM_MIN_HZ * Hz)

def _vis_syn(src, targets):
    s = Synapses(src, neu, 'w : volt', on_pre='v += w', delay=params['t_dly'])
    s.connect(i=np.arange(len(targets)), j=np.array(targets))
    s.w = params['w_syn'] * params['f_poi']
    return s

vis_syn_l = _vis_syn(vis_group_l, lam_left_ids)  if n_lam_l else None
vis_syn_r = _vis_syn(vis_group_r, lam_right_ids) if n_lam_r else None
vis_syn_b = _vis_syn(vis_group_b, lam_both_ids)  if n_lam_b else None

_vis_objects = [vis_group_l, vis_group_r, vis_group_b]
if vis_syn_l: _vis_objects.append(vis_syn_l)
if vis_syn_r: _vis_objects.append(vis_syn_r)
if vis_syn_b: _vis_objects.append(vis_syn_b)

net = Network(neu, syn, spk_mon, asc_group, asc_syn, *poi_circuit, *_vis_objects)
print(f"  Network ready ({n_neurons:,} neurons)  [{_elapsed(_t_brian)}]")
print(f"  Will run interleaved: {N_DECISIONS_TOTAL} × {int(DECISION_INTERVAL*1000)} ms steps")

# ═══════════════════════════════════════════════════════════════════════════════
#  6. BUILD FLYGYM SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding simulation ...")
arena = OdorArena(
    odor_source=FOOD_POS[np.newaxis],
    peak_odor_intensity=np.array([[1.0, 0.0]]),   # dummy - channeled odor handles steering
    diffuse_func=lambda x: x**-2,
    marker_colors=[],
)

# Food drop
food_body = arena.root_element.worldbody.add(
    "body", name="food_drop", pos=[FOOD_POS[0], FOOD_POS[1], -0.25], mocap=True
)
food_body.add("geom", type="sphere", size=[0.55],
              rgba=[0.72, 0.38, 0.08, 0.92], contype=0, conaffinity=0)


# Proboscis probe (bright blue capsule)
probe_body = arena.root_element.worldbody.add(
    "body", name="proboscis_probe", pos=[0, 0, 50], mocap=True
)
probe_body.add("geom", type="capsule", size=[0.055, 0.35],
               rgba=[0.1, 0.5, 1.0, 1.0], contype=0, conaffinity=0)

# Zigzag / tunnel layout - two walls create a corridor the fly must navigate through.
# Wall 1: x=8,  y=-15..10  gap at y>=10  (fly goes north over this wall)
# Wall 2: x=14, y=6..16    gap at y<=6   (fly comes south through tunnel gap)
# Tunnel: x=8..14, y=6..10 (corridor between the two walls)
# solimp/solref softening prevents BADQACC with adhesion actuators on tarsi.
# base z=-1mm (pos_z=3.0, half=4.0) truly blocks fly body z=-0.36..-0.07mm.
arena.root_element.worldbody.add(
    "geom", name="wall1",
    type="box", pos=[8.0, -2.5, 3.0], size=[0.3, 12.5, 4.0],
    rgba=[0.15, 0.10, 0.05, 1.0], contype=1, conaffinity=1,
    solimp="0.9 0.999 0.001 0.5 2", solref="0.02 1",
)
arena.root_element.worldbody.add(
    "geom", name="wall2",
    type="box", pos=[14.0, 11.0, 3.0], size=[0.3, 5.0, 4.0],
    rgba=[0.15, 0.10, 0.05, 1.0], contype=1, conaffinity=1,
    solimp="0.9 0.999 0.001 0.5 2", solref="0.02 1",
)

contact_sensor_placements = [
    f"{leg}{seg}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]
fly = Fly(
    spawn_pos=(0, 0, 0.2),
    spawn_orientation=(0, 0, 0.588),
    contact_sensor_placements=contact_sensor_placements,
    enable_olfaction=True, enable_adhesion=True, draw_adhesion=False,
    enable_vision=True,
)
cam_iso = YawOnlyCamera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_top_right",
    targeted_fly_names=fly.name,
    timestamp_text=False, play_speed_text=False,
    play_speed=PLAY_SPEED, fps=FPS_V,
)
cam_top = YawOnlyCamera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_top_zoomout",
    targeted_fly_names=fly.name,
    timestamp_text=False, play_speed_text=False,
    play_speed=PLAY_SPEED, fps=FPS_V,
)
# Third-person "cat cam" — 4mm behind, 3.5mm above lower back, yaws with fly.
# Parameters confirmed in tests/test_back_camera.py.
cam_back = YawOnlyCamera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_back_close",
    targeted_fly_names=fly.name,
    timestamp_text=False, play_speed_text=False,
    play_speed=PLAY_SPEED, fps=FPS_V,
    camera_parameters={
        "class": "nmf",
        "mode":  "track",
        "ipd":   0.068,
        "pos":   [-4, 0, 3.5],
        "euler": [1.1, 0.0, -1.57],
    },
)
sim = HybridTurningController(
    fly=fly, cameras=[cam_iso, cam_top, cam_back], arena=arena, timestep=PHYSICS_TIMESTEP,
)
obs, _ = sim.reset()

all_qpos       = list(sim.physics.named.data.qpos.axes[0].names)
probe_mocap_id = sim.physics.named.model.body_mocapid["proboscis_probe"]
root_joint     = next(
    (n for n in all_qpos if "dust" not in n
     and hasattr(sim.physics.named.data.qpos[n], "__len__")
     and len(sim.physics.named.data.qpos[n]) == 7),
    None,
)

def _fly_head_pos():
    try:
        return np.array(sim.physics.named.data.xpos["0/Haustellum"])
    except Exception:
        pass
    return np.array(obs["fly"][0, :3]) + np.array([1.0, 0.0, 0.0])

def _update_probe(fraction, head_pos, head_fwd):
    f = float(np.clip(fraction, 0, 1))
    if f < 0.01:
        sim.physics.data.mocap_pos[probe_mocap_id] = [0, 0, 50]
        return
    endpoint = head_pos + head_fwd * f * 0.9
    midpoint  = (head_pos + endpoint) / 2.0
    sim.physics.data.mocap_pos[probe_mocap_id] = midpoint
    direction = endpoint - head_pos
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-6:
        return
    direction /= d_norm
    z    = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction)
    an   = np.linalg.norm(axis)
    if an > 1e-6:
        axis  /= an
        angle  = np.arccos(np.clip(np.dot(z, direction), -1, 1))
        q      = Rotation.from_rotvec(axis * angle).as_quat()
        sim.physics.data.mocap_quat[probe_mocap_id] = [q[3], q[0], q[1], q[2]]

# ── Output versioning (shared by video + data file) ───────────────────────────
sim_dir  = Path(__file__).parent / "simulations"
sim_dir.mkdir(exist_ok=True)
_versions = [int(m.group(1)) for f in sim_dir.glob("v*_*.mp4")
             if (m := re.match(r"v(\d+)_", f.name))]
next_v   = max(_versions, default=0) + 1
print(f"  Output version: v{next_v}")

# ═══════════════════════════════════════════════════════════════════════════════
#  7. FLYVIS VISUAL NETWORK — load once, carry state across the loop
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading flyvis pretrained network (Lappalainen et al. 2024) ...")
import flyvis
from flyvis import NetworkView

_nv         = NetworkView(flyvis.results_dir / 'flow/0000/000')
_fv_network = _nv.init_network(checkpoint='best')
_fv_rm      = _load_retina_mapper()
_fv_t5a_idx = _fv_network.stimulus.layer_index['T5a']
_fv_t5b_idx = _fv_network.stimulus.layer_index['T5b']

# Compute initial steady state (1 s of grey input = 40 frames at 25 ms)
print("  Computing flyvis steady state (1s grey) ...")
_fv_network.eval()
for _p in _fv_network.parameters():
    _p.requires_grad = False
_grey_t = torch.ones((2, 1, 1, 721), dtype=torch.float32) * 0.5
with torch.no_grad():
    _fv_network.stimulus.zero(2, 1)
    _fv_network.stimulus.add_input(_grey_t)
    _fv_state = _fv_network.forward(_fv_network.stimulus(), FLYVIS_DT, state=None, as_states=True)[-1]
    for _ in range(39):
        _fv_network.stimulus.zero(2, 1)
        _fv_network.stimulus.add_input(_grey_t)
        _fv_state = _fv_network.forward(_fv_network.stimulus(), FLYVIS_DT, state=_fv_state, as_states=True)[-1]
print("  flyvis ready. T5a/T5b cells:", len(_fv_t5a_idx), "/", len(_fv_t5b_idx))

# ── Pre-compute channeled odor field (Dijkstra path-distance) ────────────────
print("\nBuilding channeled odor field (Dijkstra) ...")
_odor_field, _odor_xs, _odor_ys, _odor_blocked = build_odor_field(FOOD_POS[:2])
print(f"  Grid: {len(_odor_xs)} x {len(_odor_ys)} cells at {GRID_RES}mm/cell")
print(f"  Odor at spawn (0,0): {lookup_odor(0.0, 0.0, _odor_field, _odor_xs, _odor_ys):.3f}")
print(f"  Odor at food  ({FOOD_POS[0]},{FOOD_POS[1]}): {lookup_odor(FOOD_POS[0], FOOD_POS[1], _odor_field, _odor_xs, _odor_ys):.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  8. PHYSICS LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning {PHYS_DURATION_S:.0f}s interleaved brain+physics ...")
print(f"  {N_DECISIONS_TOTAL} × {int(DECISION_INTERVAL*1000)} ms steps  |  proprio feedback: ascending @ {SENSORY_STIM_RATE} Hz (dynamic)")
_t_phys = time.time()
phase       = "walk"
feed_timer  = 0.0
freeze_qpos = None
prev_joints = obs.get("joints", None)   # initial joint angles for velocity computation
if prev_joints is not None:
    prev_joints = np.array(prev_joints, dtype=float)

# Per-step recording arrays (fed into HDF5 after the loop)
rec_asc_rate   = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_lr_diff    = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_dist       = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_dn_left    = np.zeros(N_DECISIONS_TOTAL, dtype=np.int32)
rec_dn_right   = np.zeros(N_DECISIONS_TOTAL, dtype=np.int32)
rec_ctrl_left  = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_ctrl_right = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_vis_left   = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_vis_right  = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_loom_sig_l = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_loom_sig_r = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_loom_bias  = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_fly_x      = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_fly_y      = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_fly_head   = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_odor_left  = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
rec_odor_right = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)

# Per-decision records for brain overlay (one entry per decision step)
dec_odor_norm  = []
dec_is_feeding = []

# flyvis reflex state — carried across steps
_loom_persist = 0.0    # decaying T5-asymmetry bias
_loom_sig_l   = 0.0

# Navigation milestone tracking — prints once per milestone, never repeats
_nav_milestone = "start"   # states: start, north, wall1, gap1, tunnel, gap2, past_wall2

# Per-step odor asymmetry recording (positive = odor pulling right, negative = left)
rec_odor_asym  = np.zeros(N_DECISIONS_TOTAL, dtype=np.float32)
_loom_sig_r   = 0.0

for t in range(N_DECISIONS_TOTAL):
    phys_t   = t * DECISION_INTERVAL
    _t_step  = time.time()

    # ── Proprioceptive feedback: leg kinematics → ascending neuron rate ───────
    cur_joints = obs.get("joints", None)
    if cur_joints is not None and prev_joints is not None:
        delta       = np.array(cur_joints, dtype=float) - prev_joints
        velocity    = float(np.linalg.norm(delta)) / DECISION_INTERVAL        # rad/s
        activity    = float(np.clip(velocity / MAX_JOINT_VELOCITY, 0.0, 1.0))
        asc_rate    = SENSORY_STIM_RATE * (PROPRIO_MIN + (1.0 - PROPRIO_MIN) * activity)
    else:
        asc_rate = float(SENSORY_STIM_RATE)
    if cur_joints is not None:
        prev_joints = np.array(cur_joints, dtype=float)

    # ── Update ascending rate (no recompilation — see tests/test_poisson_rate_update.py)
    asc_group.rates = asc_rate * Hz

    # ── Visual rate: compound-eye luminance → LA>ME lamina neuron firing rate ──
    # obs["vision"] shape (2, 721, 2), float32, range ~[0, 1]
    _vis = obs.get("vision", None)
    if _vis is not None:
        _lum_l = float(np.clip(_vis[0].mean(), 0.0, 1.0))
        _lum_r = float(np.clip(_vis[1].mean(), 0.0, 1.0))
        _lum_b = (_lum_l + _lum_r) / 2.0
        _vis_rate_l = VIS_STIM_MIN_HZ + _lum_l * (VIS_STIM_MAX_HZ - VIS_STIM_MIN_HZ)
        _vis_rate_r = VIS_STIM_MIN_HZ + _lum_r * (VIS_STIM_MAX_HZ - VIS_STIM_MIN_HZ)
        _vis_rate_b = VIS_STIM_MIN_HZ + _lum_b * (VIS_STIM_MAX_HZ - VIS_STIM_MIN_HZ)
        vis_group_l.rates = _vis_rate_l * Hz
        vis_group_r.rates = _vis_rate_r * Hz
        vis_group_b.rates = _vis_rate_b * Hz
        # flyvis biological reflex: obs["vision"] -> T5 motion detectors -> turn bias
        # T5 = OFF-pathway; asymmetry (left eye > right eye) -> steer right
        _vis_gray = _vis.max(axis=2).astype(np.float32)       # (2, 721)
        _vis_mapped = _fv_rm.flygym_to_flyvis(_vis_gray)      # reorder to flyvis hex convention
        _frame_t = torch.tensor(_vis_mapped, dtype=torch.float32).unsqueeze(1).unsqueeze(2)  # (2,1,1,721)
        with torch.no_grad():
            _fv_network.stimulus.zero(2, 1)
            _fv_network.stimulus.add_input(_frame_t)
            _fv_states = _fv_network.forward(_fv_network.stimulus(), FLYVIS_DT, state=_fv_state, as_states=True)
            _fv_state  = _fv_states[-1]
            _act = _fv_state.nodes.activity                   # (2, n_cells)
            _loom_sig_l = float(_act[0, _fv_t5a_idx].abs().mean() + _act[0, _fv_t5b_idx].abs().mean())
            _loom_sig_r = float(_act[1, _fv_t5a_idx].abs().mean() + _act[1, _fv_t5b_idx].abs().mean())
        _t5_asym = _loom_sig_l - _loom_sig_r
        _loom_new    = -FLYVIS_T5_GAIN * _t5_asym
        _loom_persist = float(np.clip(
            _loom_persist * FLYVIS_DECAY + _loom_new * (1.0 - FLYVIS_DECAY),
            -FLYVIS_BIAS_MAX, FLYVIS_BIAS_MAX))
    else:
        _vis_rate_l = _vis_rate_r = VIS_STIM_MIN_HZ
        _loom_persist = _loom_persist * FLYVIS_DECAY

    _t_proprio = time.time()

    # ── Brian2: run 25 ms ─────────────────────────────────────────────────────
    if t == 0:
        print("  [step 0] First net.run() — C++ compilation happens here if not cached ...")
    prev_n      = int(spk_mon.num_spikes)
    net.run(DECISION_INTERVAL * second)
    _t_brian_done = time.time()
    new_i       = np.asarray(spk_mon.i)[prev_n:]
    left_count  = int(np.isin(new_i, dn_left_ids).sum())
    right_count = int(np.isin(new_i, dn_right_ids).sum())
    lr_diff_t   = (left_count - right_count) / (left_count + right_count + 1e-6)

    pos    = obs["fly"][0, :2]
    dist   = float(np.linalg.norm(pos - FOOD_POS[:2]))

    # ── Channeled odor: Dijkstra path-distance field, antennas at fly heading ──
    if root_joint:
        _qpos    = sim.physics.named.data.qpos[root_joint]
        _pos_x   = float(_qpos[0])
        _pos_y   = float(_qpos[1])
        _qw      = float(_qpos[3])
        _qz      = float(_qpos[6])
        _heading = float(np.arctan2(2.0 * _qw * _qz, 1.0 - 2.0 * _qz**2))
    else:
        _pos_x, _pos_y = float(pos[0]), float(pos[1])
        _heading = 0.0
    _right_x =  np.sin(_heading)
    _right_y = -np.cos(_heading)
    left_odor  = lookup_odor(_pos_x - ANT_SEP * _right_x,
                             _pos_y - ANT_SEP * _right_y,
                             _odor_field, _odor_xs, _odor_ys)
    right_odor = lookup_odor(_pos_x + ANT_SEP * _right_x,
                             _pos_y + ANT_SEP * _right_y,
                             _odor_field, _odor_xs, _odor_ys)
    total_odor = left_odor + right_odor
    _lr_asym   = (right_odor - left_odor) / (total_odor + 1e-9)   # + = pull right, - = pull left
    _pull_dir  = "right" if _lr_asym > 0.02 else ("left" if _lr_asym < -0.02 else "straight")

    # ── navigation milestone detection (prints once per milestone) ────────────
    if _nav_milestone == "start" and _pos_y > 3.0:
        _nav_milestone = "north"
        print(f"  [NAV] Walking north  step={t}  x={_pos_x:.1f} y={_pos_y:.1f}"
              f"  odor L={left_odor:.2f} R={right_odor:.2f}  pull={_pull_dir}")

    elif _nav_milestone == "north" and _pos_x >= 7.0:
        _nav_milestone = "wall1"
        print(f"  [NAV] Approaching wall 1  step={t}  x={_pos_x:.1f} y={_pos_y:.1f}"
              f"  odor L={left_odor:.2f} R={right_odor:.2f}  pull={_pull_dir}")

    elif _nav_milestone == "wall1" and _pos_y >= 10.0:
        _nav_milestone = "gap1"
        print(f"  [NAV] *** Found wall 1 gap (y>=10)!  step={t}  x={_pos_x:.1f} y={_pos_y:.2f}"
              f"  odor L={left_odor:.2f} R={right_odor:.2f}  pull={_pull_dir}")

    elif _nav_milestone == "gap1" and _pos_x > 8.2 and 6.0 <= _pos_y <= 10.5:
        _nav_milestone = "tunnel"
        print(f"  [NAV] Entered tunnel (x=8..14, y=6..10)  step={t}  x={_pos_x:.1f} y={_pos_y:.1f}"
              f"  odor L={left_odor:.2f} R={right_odor:.2f}  pull={_pull_dir}")

    elif _nav_milestone == "tunnel" and _pos_y <= 6.0:
        _nav_milestone = "gap2"
        print(f"  [NAV] *** Found wall 2 gap (y<=6)!  step={t}  x={_pos_x:.1f} y={_pos_y:.2f}"
              f"  odor L={left_odor:.2f} R={right_odor:.2f}  pull={_pull_dir}")

    elif _nav_milestone == "gap2" and _pos_x >= 14.0:
        _nav_milestone = "past_wall2"
        print(f"  [NAV] *** Past wall 2 - heading to food!  step={t}  x={_pos_x:.1f} y={_pos_y:.1f}"
              f"  dist={dist:.1f}mm  odor L={left_odor:.2f} R={right_odor:.2f}  pull={_pull_dir}")

    # ── periodic odor + position status every 20 steps ───────────────────────
    if t % 20 == 0:
        print(f"  [ODR] step={t:03d}  pos=({_pos_x:.1f},{_pos_y:.1f})"
              f"  heading={np.degrees(_heading):.0f}deg"
              f"  odor L={left_odor:.3f} R={right_odor:.3f}  asym={_lr_asym:+.3f}  pull={_pull_dir}"
              f"  dist={dist:.1f}mm  nav={_nav_milestone}")

    # ── phase transitions ────────────────────────────────────────────────────
    if phase == "walk" and dist < FEED_DIST:
        phase = "feed_extend"
        feed_timer = 0.0
        if root_joint:
            freeze_qpos = sim.physics.named.data.qpos[root_joint].copy()
        print(f"  FEEDING START  t={phys_t:.2f}s  dist={dist:.2f}mm")

    if phase == "feed_extend" and feed_timer >= 0.5:
        phase = "feed_eat"

    if phase == "feed_eat" and feed_timer >= FEED_DUR:
        phase = "feed_retract"
        feed_timer = 0.0

    if phase == "feed_retract" and feed_timer >= 0.5:
        phase = "walk"
        freeze_qpos = None
        print(f"  RESUME WALK  t={phys_t:.2f}s")

    # ── proboscis probe ───────────────────────────────────────────────────────
    head_pos = _fly_head_pos()
    to_food  = FOOD_POS - head_pos
    fwd      = to_food / (np.linalg.norm(to_food) + 1e-6)

    if phase == "feed_extend":
        _update_probe(feed_timer / 0.5, head_pos, fwd)
    elif phase == "feed_eat":
        _update_probe(1.0 + 0.05 * np.sin(feed_timer * 8 * np.pi), head_pos, fwd)
    elif phase == "feed_retract":
        _update_probe(1.0 - feed_timer / 0.5, head_pos, fwd)
    else:
        _update_probe(0.0, head_pos, fwd)

    # ── control signal ────────────────────────────────────────────────────────
    if phase == "walk":
        # Odor gradient for primary steering (_lr_asym already computed above)
        odor_turn = np.tanh(_lr_asym * 20.0) * ODOR_TURN_K
        # DN left/right asymmetry adds subtle brain-grounded variation
        dn_bias   = float(lr_diff_t) * 0.15
        turn_bias = odor_turn + dn_bias + _loom_persist
        ctrl = np.array([
            float(np.clip(WALK_AMP + turn_bias, 0.1, 1.0)),
            float(np.clip(WALK_AMP - turn_bias, 0.1, 1.0)),
        ])
    else:
        ctrl = np.array([0.0, 0.0])

    # ── microsteps ────────────────────────────────────────────────────────────
    _t_physics_start = time.time()
    for _ in range(PHYSICS_STEPS_PER_DECISION):
        obs, _, _, _, _ = sim.step(ctrl)
        if freeze_qpos is not None and root_joint:
            sim.physics.named.data.qpos[root_joint] = freeze_qpos
            sim.physics.named.data.qvel[root_joint] = 0.0
        sim.render()
    _t_physics_done = time.time()

    # ── record state ─────────────────────────────────────────────────────────
    # total_odor ~ 2 at far distance, ~250 near food; normalise at 100 (≈half-max)
    dec_odor_norm.append(float(np.clip(total_odor / 100.0, 0.0, 1.0)))
    dec_is_feeding.append(
        1.0 if phase in ("feed_extend", "feed_eat", "feed_retract") else 0.0
    )

    if phase in ("feed_extend", "feed_eat", "feed_retract"):
        feed_timer += DECISION_INTERVAL

    # ── per-step timing log (every step for first 5, then every 10) ──────────
    ms_proprio = (_t_proprio      - _t_step)         * 1000
    ms_brian   = (_t_brian_done   - _t_proprio)      * 1000
    ms_physics = (_t_physics_done - _t_physics_start) * 1000
    ms_total   = (_t_physics_done - _t_step)         * 1000
    if t < 5 or t % 10 == 0:
        elapsed_so_far = time.time() - _t_phys
        eta_s = (elapsed_so_far / (t + 1)) * (N_DECISIONS_TOTAL - t - 1)
        dn_tag = f"DN L{left_count}/R{right_count} lr={lr_diff_t:+.2f}"
        print(
            f"  step {t:03d}/{N_DECISIONS_TOTAL}  t={phys_t:.2f}s  [{phase}]"
            f"  proprio={ms_proprio:.0f}ms"
            f"  brian={ms_brian:.0f}ms"
            f"  physics={ms_physics:.0f}ms"
            f"  step_total={ms_total:.0f}ms"
            f"  asc={asc_rate:.0f}Hz  {dn_tag}"
            f"  dist={dist:.1f}mm"
            f"  ETA={eta_s/60:.1f}min"
        )

    # ── per-step data recording ───────────────────────────────────────────────
    rec_asc_rate[t]   = asc_rate
    rec_lr_diff[t]    = lr_diff_t
    rec_dist[t]       = dist
    rec_dn_left[t]    = left_count
    rec_dn_right[t]   = right_count
    rec_ctrl_left[t]  = ctrl[0]
    rec_ctrl_right[t] = ctrl[1]
    rec_vis_left[t]   = _vis_rate_l
    rec_vis_right[t]  = _vis_rate_r
    rec_loom_sig_l[t] = _loom_sig_l
    rec_loom_sig_r[t] = _loom_sig_r
    rec_loom_bias[t]  = _loom_persist
    rec_odor_left[t]  = left_odor
    rec_odor_right[t] = right_odor
    rec_odor_asym[t]  = _lr_asym
    if root_joint:
        _qpos = sim.physics.named.data.qpos[root_joint]
        rec_fly_x[t] = float(_qpos[0])
        rec_fly_y[t] = float(_qpos[1])
        # yaw from quaternion (qw, qx, qy, qz at indices 3..6)
        _qw, _qz = float(_qpos[3]), float(_qpos[6])
        rec_fly_head[t] = float(np.arctan2(2.0*_qw*_qz, 1.0 - 2.0*_qz*_qz))

frames_iso  = cam_iso._frames
frames_top  = cam_top._frames
frames_back = cam_back._frames
n_video_frames = len(frames_iso)
raw_trains = spk_mon.spike_trains()
n_spiking  = sum(1 for v in raw_trains.values() if len(v) > 0)
print(f"  {n_video_frames} fly frames captured  |  {n_spiking:,} neurons fired  [{_elapsed(_t_phys)}]")

# ═══════════════════════════════════════════════════════════════════════════════
#  5b. COMPUTE BRAIN GLOW FRAMES (from accumulated spike trains)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing brain glow frames ...")
_t_glow = time.time()
n_brain_frames  = int(np.ceil(BRAIN_DURATION_MS / FRAME_DT_MS))
decay_per_frame = np.exp(-FRAME_DT_MS / DECAY_TAU_MS)

all_spikes = []
for bidx_raw, times in raw_trains.items():
    bidx = int(bidx_raw)
    if bidx in dense_map and len(times) > 0:
        d = dense_map[bidx]
        for t_sp in times:
            all_spikes.append((float(t_sp / ms), d))
all_spikes.sort()

glow        = np.zeros(n_valid, dtype=np.float32)
frame_glows = np.zeros((n_brain_frames, n_valid), dtype=np.float32)
spike_ptr   = 0
n_sp        = len(all_spikes)

for f in range(n_brain_frames):
    t_end = (f + 1) * FRAME_DT_MS
    glow *= decay_per_frame
    while spike_ptr < n_sp and all_spikes[spike_ptr][0] < t_end:
        glow[all_spikes[spike_ptr][1]] += 1.0
        spike_ptr += 1
    frame_glows[f] = glow

max_glow = frame_glows.max()
if max_glow > 0:
    frame_glows /= max_glow
print(f"  {n_brain_frames} brain frames  |  peak raw glow: {max_glow:.1f}  [{_elapsed(_t_glow)}]")

# ═══════════════════════════════════════════════════════════════════════════════
#  8. INTERPOLATE DECISION STATE → PER-VIDEO-FRAME
# ═══════════════════════════════════════════════════════════════════════════════
dec_idx    = np.linspace(0, len(dec_odor_norm) - 1, n_video_frames)
vf_odor    = np.interp(dec_idx, np.arange(len(dec_odor_norm)),    dec_odor_norm)
vf_feeding = np.interp(dec_idx, np.arange(len(dec_is_feeding)), dec_is_feeding)

# ═══════════════════════════════════════════════════════════════════════════════
#  9. PRE-RENDER BRAIN PANELS  (one per video frame — 4 colour-coded layers)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nRendering {n_video_frames} brain panels (4-layer, colour-coded) ...")
_t_render = time.time()

# ── colormaps per circuit ─────────────────────────────────────────────────────
cmap_lif = LinearSegmentedColormap.from_list(
    "lif", ["#0a0e14", "#103050", "#00aacc", "#88ddff", "#ffffff"])
cmap_dn = LinearSegmentedColormap.from_list(
    "dn",  ["#050f07", "#00cc44", "#33ff77", "#88ffaa", "#ffffff"])
cmap_olf = LinearSegmentedColormap.from_list(
    "olf", ["#0f0510", "#3a0a30", "#cc2288", "#ff55bb", "#ffccee"])
cmap_sez = LinearSegmentedColormap.from_list(
    "sez", ["#100500", "#301500", "#cc5500", "#ff9944", "#ffeecc"])
cmap_vis = LinearSegmentedColormap.from_list(
    "vis", ["#0a0a00", "#252000", "#bbaa00", "#ffee00", "#ffffff"])

fig_b = plt.figure(figsize=(BRAIN_PANEL_W / 100, BRAIN_PANEL_H / 100),
                   dpi=100, facecolor="black")
ax_b = fig_b.add_axes([0, 0.10, 1, 0.90])
ax_b.set_facecolor("black")
ax_b.set_aspect("equal")
ax_b.axis("off")
xpad = (px.max() - px.min()) * 0.03
zpad = (pz.max() - pz.min()) * 0.03
ax_b.set_xlim(px.min() - xpad, px.max() + xpad)
ax_b.set_ylim(pz.min() - zpad, pz.max() + zpad)

# Layer 0 — static anatomical background (all neurons, always visible)
ax_b.scatter(px, pz, s=1.5, c="#1a3a5c",
             alpha=1.0, linewidths=0, rasterized=True, zorder=1)

# Layer 1 — LIF spike activity (all neurons, cyan-white)
g0   = frame_glows[0]
sc_lif = ax_b.scatter(px, pz,
                      s=DOT_BG_SIZE + g0 * DOT_LIF_SIZE,
                      c=g0, cmap=cmap_lif, vmin=0, vmax=1,
                      alpha=0.7, linewidths=0, rasterized=True, zorder=2)

# Layer 2 — Descending neurons (bright green — locomotion)
# vmax=0.25 so even small LIF values map to bright green (not dark)
if len(dn_dense) > 0:
    g_dn = g0[dn_dense]
    sc_dn = ax_b.scatter(px[dn_dense], pz[dn_dense],
                         s=DOT_BG_SIZE + g_dn * DOT_DN_SIZE,
                         c=g_dn, cmap=cmap_dn, vmin=0, vmax=0.25,
                         alpha=0.75, linewidths=0, rasterized=True, zorder=3)
else:
    sc_dn = None

# Layer 3 — Olfactory neurons (pink)
# vmax=0.25 + behavioral floor ensures visibility when odor detected
if len(olfactory_dense) > 0:
    g_olf = g0[olfactory_dense]
    sc_olf = ax_b.scatter(px[olfactory_dense], pz[olfactory_dense],
                          s=DOT_BG_SIZE + g_olf * DOT_OLF_SIZE,
                          c=g_olf, cmap=cmap_olf, vmin=0, vmax=0.25,
                          alpha=0.08, linewidths=0, rasterized=True, zorder=4)
else:
    sc_olf = None

# Layer 4 — SEZ / feeding neurons (orange)
# vmax=0.25 + behavioral floor ensures visibility during feeding
if len(sez_dense) > 0:
    g_sez = g0[sez_dense]
    sc_sez = ax_b.scatter(px[sez_dense], pz[sez_dense],
                          s=DOT_BG_SIZE + g_sez * DOT_SEZ_SIZE,
                          c=g_sez, cmap=cmap_sez, vmin=0, vmax=0.25,
                          alpha=0.08, linewidths=0, rasterized=True, zorder=5)
else:
    sc_sez = None

# Layer 5 — LA>ME lamina / visual neurons (yellow)
if len(lam_dense) > 0:
    g_vis = g0[lam_dense]
    sc_vis = ax_b.scatter(px[lam_dense], pz[lam_dense],
                          s=DOT_BG_SIZE + g_vis * DOT_VIS_SIZE,
                          c=g_vis, cmap=cmap_vis, vmin=0, vmax=0.25,
                          alpha=0.08, linewidths=0, rasterized=True, zorder=6)
else:
    sc_vis = None

# ── legend (bottom strip) ─────────────────────────────────────────────────────
# background strip
ax_leg = fig_b.add_axes([0, 0.0, 1, 0.10])
ax_leg.set_facecolor("#050810")
ax_leg.axis("off")

ax_leg.text(0.01, 0.50, "●", color="#4ab8cc", fontsize=11, va="center", ha="left",
            transform=ax_leg.transAxes)
ax_leg.text(0.06, 0.50, "LIF activity", color="#4ab8cc", fontsize=8,
            va="center", ha="left", transform=ax_leg.transAxes)

ax_leg.text(0.21, 0.50, "●", color="#33ff88", fontsize=11, va="center", ha="left",
            transform=ax_leg.transAxes)
ax_leg.text(0.26, 0.50, "locomotion (DN)", color="#33ff88", fontsize=8,
            va="center", ha="left", transform=ax_leg.transAxes)

ax_leg.text(0.44, 0.50, "●", color="#ff55bb", fontsize=11, va="center", ha="left",
            transform=ax_leg.transAxes)
ax_leg.text(0.49, 0.50, "olfactory", color="#ff55bb", fontsize=8,
            va="center", ha="left", transform=ax_leg.transAxes)

ax_leg.text(0.62, 0.50, "●", color="#ff9944", fontsize=11, va="center", ha="left",
            transform=ax_leg.transAxes)
ax_leg.text(0.67, 0.50, "SEZ/feeding", color="#ff9944", fontsize=8,
            va="center", ha="left", transform=ax_leg.transAxes)

ax_leg.text(0.81, 0.50, "●", color="#ffee00", fontsize=11, va="center", ha="left",
            transform=ax_leg.transAxes)
ax_leg.text(0.86, 0.50, "visual (LA>ME)", color="#ffee00", fontsize=8,
            va="center", ha="left", transform=ax_leg.transAxes)

# ── time + state label ────────────────────────────────────────────────────────
t_txt  = fig_b.text(0.97, 0.98, "", ha="right", va="top",
                    fontsize=8, color="#888888")
st_txt = fig_b.text(0.03, 0.98, "", ha="left",  va="top",
                    fontsize=9, color="white", fontweight="bold")

# ── render loop ───────────────────────────────────────────────────────────────
# Video frame i → physics time → brain time (1:1, no modulo)
phys_t_per_frame = PHYS_DURATION_S / max(n_video_frames, 1)
brain_panels = []

for i in range(n_video_frames):
    phys_t     = i * phys_t_per_frame
    brain_t_ms = phys_t * 1000.0                                # 1:1 with physics
    bf         = int(np.clip(brain_t_ms / FRAME_DT_MS, 0, n_brain_frames - 1))

    g_all = frame_glows[bf]

    # ── LIF layer (all neurons) ───────────────────────────────────────────────
    sc_lif.set_array(g_all)
    sc_lif.set_sizes(DOT_BG_SIZE + g_all * DOT_LIF_SIZE)

    # ── DN layer (bright green — fixed alpha, individual spike pattern) ─────────
    # vmax=0.25 on the scatter means even small glow values render bright green.
    # Alpha fixed at 0.75 so DNs are always clearly visible.
    if sc_dn is not None and len(dn_dense) > 0:
        g_dn = g_all[dn_dense]
        sc_dn.set_array(g_dn)
        sc_dn.set_sizes(DOT_BG_SIZE + g_dn * DOT_DN_SIZE)
        sc_dn.set_alpha(0.75)

    # ── Olfactory layer (pink — real Brian2 spike pattern, alpha ∝ odor) ────────
    # Neurons were stimulated at 80 Hz in Brian2, so g_all contains real individual
    # spike glows. Alpha scales with odor intensity so the circuit fades in/out.
    if sc_olf is not None and len(olfactory_dense) > 0:
        olf_boost = float(vf_odor[i])
        g_olf = g_all[olfactory_dense]
        sc_olf.set_array(g_olf)
        sc_olf.set_sizes(DOT_BG_SIZE + g_olf * DOT_OLF_SIZE)
        sc_olf.set_alpha(float(np.clip(0.05 + olf_boost * 0.85, 0.05, 0.90)))

    # ── SEZ layer (orange — real Brian2 spike pattern, alpha ∝ feeding) ─────────
    if sc_sez is not None and len(sez_dense) > 0:
        sez_boost = float(vf_feeding[i])
        g_sez = g_all[sez_dense]
        sc_sez.set_array(g_sez)
        sc_sez.set_sizes(DOT_BG_SIZE + g_sez * DOT_SEZ_SIZE)
        sc_sez.set_alpha(float(np.clip(0.05 + sez_boost * 0.85, 0.05, 0.90)))

    # ── Visual layer (yellow — LA>ME lamina, alpha ∝ mean luminance) ─────────
    if sc_vis is not None and len(lam_dense) > 0:
        # Use mean of rec_vis_left/right at this video frame (interpolated)
        dec_i = int(np.clip(i * N_DECISIONS_TOTAL / max(n_video_frames, 1),
                            0, N_DECISIONS_TOTAL - 1))
        vis_lum = float(np.clip(
            (rec_vis_left[dec_i] + rec_vis_right[dec_i]) / 2.0 / VIS_STIM_MAX_HZ,
            0.0, 1.0))
        g_vis = g_all[lam_dense]
        sc_vis.set_array(g_vis)
        sc_vis.set_sizes(DOT_BG_SIZE + g_vis * DOT_VIS_SIZE)
        sc_vis.set_alpha(float(np.clip(0.05 + vis_lum * 0.80, 0.05, 0.85)))

    # ── labels ────────────────────────────────────────────────────────────────
    t_txt.set_text(f"t = {brain_t_ms / 1000:.2f} s")
    if vf_feeding[i] > 0.1:
        st_txt.set_text("feeding")
        st_txt.set_color("#ff9944")
    elif vf_odor[i] > 0.05:
        st_txt.set_text("odor detected")
        st_txt.set_color("#ff55bb")
    else:
        st_txt.set_text("walking")
        st_txt.set_color("#33ff88")

    fig_b.canvas.draw()
    buf = np.frombuffer(fig_b.canvas.buffer_rgba(), dtype=np.uint8)
    brain_panels.append(buf.reshape(BRAIN_PANEL_H, BRAIN_PANEL_W, 4)[:, :, :3].copy())

    if i % 100 == 0:
        print(f"  rendered {i}/{n_video_frames} ...")

plt.close(fig_b)
print(f"  {len(brain_panels)} panels done  [{_elapsed(_t_render)}]")

# ═══════════════════════════════════════════════════════════════════════════════
#  10. WRITE SPLIT-SCREEN VIDEO
# ═══════════════════════════════════════════════════════════════════════════════
print("Writing video ...")
_t_video = time.time()
out_path = sim_dir / f"v{next_v}_brain_body_v4.mp4"

# Layout:  ROW 1 = brain panel        (BRAIN_PANEL_W × BRAIN_PANEL_H)
#          ROW 2 = iso view | top-down (each BRAIN_PANEL_W/2 wide)
#          ROW 3 = cat-cam back view   (BRAIN_PANEL_W wide — full width)
FLY_H, FLY_W = frames_iso[0].shape[:2]
fly_panel_w  = BRAIN_PANEL_W // 2          # iso + top-down each half width
fly_panel_h  = int(FLY_H * fly_panel_w / FLY_W)
if fly_panel_h % 2 != 0:
    fly_panel_h += 1

# Cat-cam row: full width, height scaled to keep original aspect ratio
BACK_H, BACK_W = frames_back[0].shape[:2]
back_panel_h = int(BACK_H * BRAIN_PANEL_W / BACK_W)
if back_panel_h % 2 != 0:
    back_panel_h += 1

total_w = BRAIN_PANEL_W
total_h = BRAIN_PANEL_H + fly_panel_h + back_panel_h

try:
    from PIL import Image as PILImage
    def _resize(frame, w, h):
        return np.array(PILImage.fromarray(frame).resize((w, h), PILImage.LANCZOS))
except ImportError:
    import cv2
    def _resize(frame, w, h):
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

writer = imageio.get_writer(
    str(out_path), fps=FPS_V, codec="libx264",
    macro_block_size=None,
    output_params=["-pix_fmt", "yuv420p", "-crf", "18"],
)
for i in range(n_video_frames):
    row1 = brain_panels[i]                                        # (480,  1280, 3)
    iso_panel  = _resize(frames_iso[i],  fly_panel_w, fly_panel_h)
    top_panel  = _resize(frames_top[i],  fly_panel_w, fly_panel_h)
    row2 = np.concatenate([iso_panel, top_panel], axis=1)         # (H,    1280, 3)
    row3 = _resize(frames_back[i], BRAIN_PANEL_W, back_panel_h)  # (H2,   1280, 3)
    combined = np.concatenate([row1, row2, row3], axis=0)
    writer.append_data(combined)
writer.close()

print(f"\nDone — {out_path}")
print(f"  Physics duration : {PHYS_DURATION_S:.0f}s")
print(f"  Brain duration   : {BRAIN_DURATION_S:.0f}s  (same — no tiling)")
print(f"  Video duration   : {n_video_frames / FPS_V:.1f}s  ({n_video_frames} frames @ {FPS_V} fps)")
print(f"  Resolution       : {total_w} × {total_h} px  (brain | iso+top-down | cat-cam)")
print(f"  Circuits shown   : DN (green)  olfactory (pink)  SEZ/feeding (orange)")
print(f"\n-- Timing breakdown ------------------------------------------")
print(f"  Network setup    : {_elapsed(_t_brian)}")
print(f"  Brain+physics    : {_elapsed(_t_phys)}  (interleaved -- {N_DECISIONS_TOTAL}x25ms Brian2 + physics)")
print(f"  Glow precompute  : {_elapsed(_t_glow)}")
print(f"  Brain rendering  : {_elapsed(_t_render)}")
print(f"  Video write      : {_elapsed(_t_video)}")
print(f"  TOTAL            : {_elapsed(T_START)}")

# ===============================================================================
#  11. SAVE SIMULATION DATA  (HDF5 — consumed by generate_plots.py)
# ===============================================================================
print("\nSaving simulation data ...")
_t_data   = time.time()
data_path = sim_dir / f"v{next_v}_data.h5"

def _circuit_spikes(trains, brian_ids):
    """Flatten spike trains for given Brian IDs into (times_s, local_idx) arrays."""
    t_list, i_list = [], []
    for local_i, bid in enumerate(brian_ids):
        if bid in trains:
            st = np.asarray(trains[bid], dtype=np.float32)  # seconds
            t_list.append(st)
            i_list.append(np.full(len(st), local_i, dtype=np.int32))
    times = np.concatenate(t_list) if t_list else np.zeros(0, dtype=np.float32)
    idxs  = np.concatenate(i_list) if i_list else np.zeros(0, dtype=np.int32)
    return times, idxs

# Subsample olfactory (2279→300) and draw a general population sample (300),
# both seeded so plots are reproducible across runs.
_rng          = np.random.default_rng(42)
olf_save_ids  = sorted(_rng.choice(olfactory_brian,
                                   size=min(300, len(olfactory_brian)),
                                   replace=False).tolist())
_circuit_set  = set(dn_left_ids + dn_right_ids + dn_both_ids
                    + olf_save_ids + sez_brian + sensory_idx)
_pop_pool     = [i for i in range(n_neurons) if i not in _circuit_set]
pop_sample_ids = sorted(_rng.choice(_pop_pool,
                                    size=min(300, len(_pop_pool)),
                                    replace=False).tolist())

with h5py.File(data_path, "w") as f:
    # -- metadata
    m = f.create_group("meta")
    m.attrs["n_steps"]           = N_DECISIONS_TOTAL
    m.attrs["decision_interval"] = DECISION_INTERVAL
    m.attrs["brain_duration_s"]  = BRAIN_DURATION_S
    m.attrs["sensory_stim_rate"] = SENSORY_STIM_RATE
    m.attrs["timestamp"]         = time.strftime("%Y-%m-%d %H:%M:%S")
    m.attrs["version"]           = next_v

    # -- behavioral timeseries (one value per 25ms decision step)
    b = f.create_group("behavior")
    b.create_dataset("asc_rate",       data=rec_asc_rate)
    b.create_dataset("lr_diff",        data=rec_lr_diff)
    b.create_dataset("dist_to_food",   data=rec_dist)
    b.create_dataset("dn_left_count",  data=rec_dn_left)
    b.create_dataset("dn_right_count", data=rec_dn_right)
    b.create_dataset("ctrl_left",      data=rec_ctrl_left)
    b.create_dataset("ctrl_right",     data=rec_ctrl_right)
    b.create_dataset("vis_rate_left",  data=rec_vis_left)
    b.create_dataset("vis_rate_right", data=rec_vis_right)
    b.create_dataset("loom_signal_l",  data=rec_loom_sig_l)
    b.create_dataset("loom_signal_r",  data=rec_loom_sig_r)
    b.create_dataset("loom_bias",      data=rec_loom_bias)
    b.create_dataset("fly_x",          data=rec_fly_x)
    b.create_dataset("fly_y",          data=rec_fly_y)
    b.create_dataset("fly_heading",    data=rec_fly_head)
    b.create_dataset("odor_left",      data=rec_odor_left)
    b.create_dataset("odor_right",     data=rec_odor_right)
    b.create_dataset("odor_asym",      data=rec_odor_asym)
    b.create_dataset("odor_norm",      data=np.array(dec_odor_norm,  dtype=np.float32))
    b.create_dataset("is_feeding",     data=np.array(dec_is_feeding, dtype=np.float32))

    # -- channeled odor field (Dijkstra grid) for trajectory overlay plots
    og = f.create_group("odor_field")
    og.create_dataset("field", data=_odor_field.astype(np.float32), compression="gzip")
    og.create_dataset("xs",    data=_odor_xs.astype(np.float32))
    og.create_dataset("ys",    data=_odor_ys.astype(np.float32))
    og.create_dataset("blocked", data=_odor_blocked.astype(np.uint8), compression="gzip")
    og.attrs["grid_res"] = GRID_RES
    og.attrs["food_x"]   = float(FOOD_POS[0])
    og.attrs["food_y"]   = float(FOOD_POS[1])

    # -- spike trains per circuit (flat times+idx arrays, gzip compressed)
    s = f.create_group("spikes")
    circuits = [
        ("dn_left",           dn_left_ids,    None),
        ("dn_right",          dn_right_ids,   None),
        ("dn_bilateral",      dn_both_ids,    None),
        ("olfactory",         olf_save_ids,   olf_save_ids),
        ("sez",               sez_brian,      None),
        ("ascending",         sensory_idx,    None),
        ("population_sample", pop_sample_ids, pop_sample_ids),
    ]
    for name, ids, save_brian_ids in circuits:
        grp = s.create_group(name)
        times, idxs = _circuit_spikes(raw_trains, ids)
        grp.create_dataset("times", data=times, compression="gzip")
        grp.create_dataset("idx",   data=idxs,  compression="gzip")
        grp.attrs["n_neurons"] = len(ids)
        grp.attrs["n_spikes"]  = len(times)
        if save_brian_ids is not None:
            grp.create_dataset("brian_ids",
                               data=np.array(save_brian_ids, dtype=np.int32))

    # -- soma positions for spatial context (dense-array coordinates)
    p = f.create_group("positions")
    for name, d_arr in [("dn", dn_dense), ("olfactory", olfactory_dense), ("sez", sez_dense)]:
        pg = p.create_group(name)
        pg.create_dataset("x", data=px[d_arr].astype(np.float32))
        pg.create_dataset("z", data=pz[d_arr].astype(np.float32))

print(f"  Saved {data_path.name}  [{_elapsed(_t_data)}]")
print(f"  Run:  python generate_plots.py {data_path}")

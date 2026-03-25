"""
Brain-body simulation v2: FlyWire LIF brain -> flygym NeuroMechFly body.

Pipeline
--------
1. Run the LIF connectome model (Brian2) with ascending neuron stimulation.
2. Bin descending neuron (DN) spike trains into 25 ms decision windows.
3. Map DN left/right firing asymmetry to HybridTurningController's 2D signal.
4. Step the flygym physics in an OdorArena (attractive food source ahead).
5. Save versioned video to simulations/.

Key differences from v1 (brain_body_simulation.py)
---------------------------------------------------
- Body: flygym NeuroMechFly v2 (87-DOF) replaces flybody (66-DOF)
- Gait:  built-in CPGNetwork + stumble/retraction rules (no hand-coded tripod)
- Control: 2D signal [left_amp, right_amp] instead of joint-level sinusoids
- Arena: OdorArena with a food source replaces flat floor
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# flygym / dm_control must be imported BEFORE load_dotenv so the renderer
# auto-detects the correct backend before .env sets MUJOCO_GL.
from flygym import Fly, YawOnlyCamera
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController

from dotenv import load_dotenv
load_dotenv()

# ── paths ─────────────────────────────────────────────────────────────────────
BRAIN_DIR = Path(__file__).parent / "brain_model"
PATH_COMP  = BRAIN_DIR / "Completeness_783.csv"
PATH_CON   = BRAIN_DIR / "Connectivity_783.parquet"

sys.path.insert(0, str(BRAIN_DIR))

# Brian2 imports (after dotenv so prefs override is safe)
from brian2 import (
    NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network,
    mV, ms, Hz, second, start_scope, prefs,
)
from model import create_model, poi, default_params

# ── simulation knobs ──────────────────────────────────────────────────────────
BRAIN_DURATION_MS  = 1000   # ms of neural simulation (= 1 s of physics)
STIM_RATE_HZ       = 150    # ascending neuron stimulation rate
DECISION_INTERVAL  = 0.025  # seconds per control decision step (25 ms)
PHYSICS_TIMESTEP   = 1e-4   # seconds (flygym default)

N_DECISIONS = int((BRAIN_DURATION_MS / 1000.0) / DECISION_INTERVAL)  # 40 bins
PHYSICS_STEPS_PER_DECISION = int(DECISION_INTERVAL / PHYSICS_TIMESTEP)  # 250

# Amplitude clamp: minimum control signal so CPG never fully stops
MIN_AMP = 0.3

# ── 1. Load connectome index ──────────────────────────────────────────────────
print(f"Loading connectome ({PATH_COMP.name}) ...")
df_comp = pd.read_csv(PATH_COMP, index_col=0)
n_neurons = len(df_comp)
flyid2i = {flyid: i for i, flyid in enumerate(df_comp.index)}
print(f"  {n_neurons:,} neurons")

# ── Sensory input: ascending neurons (VNC -> brain proprioception) ────────────
df_ann = pd.read_csv(BRAIN_DIR / "flywire_annotations.tsv", sep="\t", low_memory=False)
ascending_ids = df_ann[df_ann["super_class"] == "ascending"]["root_id"].tolist()
sensory_idx   = [flyid2i[fid] for fid in ascending_ids if fid in flyid2i]
print(f"  Stimulating {len(sensory_idx)} ascending neurons (leg proprioception -> brain)")

# ── Motor readout: descending neurons split by body side ─────────────────────
dn_df = pd.read_csv(BRAIN_DIR / "descending_neurons.csv")

dn_left_idx  = []
dn_right_idx = []
dn_both_idx  = []

for _, row in dn_df.iterrows():
    fid = row["root_id"]
    if fid not in flyid2i:
        continue
    bidx = flyid2i[fid]
    side = str(row.get("side", "")).strip().lower()
    if side == "left":
        dn_left_idx.append(bidx)
    elif side == "right":
        dn_right_idx.append(bidx)
    else:
        dn_both_idx.append(bidx)

print(f"  DN readout: {len(dn_left_idx)} left  {len(dn_right_idx)} right  "
      f"{len(dn_both_idx)} bilateral")

# ── 2. Run Brian2 brain model (single trial) ──────────────────────────────────
print(f"\nRunning LIF brain model ({BRAIN_DURATION_MS} ms) ...")
start_scope()
prefs.codegen.target = "numpy"   # no Cython / Visual Studio required

params = {
    **default_params,
    "t_run": BRAIN_DURATION_MS * ms,
    "n_run": 1,
    "r_poi": STIM_RATE_HZ * Hz,
}

neu, syn, spk_mon = create_model(PATH_COMP, PATH_CON, params)
poi_inp, neu      = poi(neu, sensory_idx, [], params)
net = Network(neu, syn, spk_mon, *poi_inp)
net.run(duration=params["t_run"])

raw_trains = spk_mon.spike_trains()
n_spiking  = sum(1 for v in raw_trains.values() if len(v) > 0)
print(f"  {n_spiking:,} neurons fired")

# ── 3. Bin DN spike trains into decision windows ──────────────────────────────
duration_s = BRAIN_DURATION_MS / 1000.0
bin_edges  = np.linspace(0.0, duration_s, N_DECISIONS + 1)

def _bin_group(indices):
    """Total spikes per decision bin summed across a neuron group."""
    total = np.zeros(N_DECISIONS, dtype=float)
    for bidx in indices:
        if bidx in raw_trains and len(raw_trains[bidx]) > 0:
            times_s = np.array(raw_trains[bidx] / second)
            counts, _ = np.histogram(times_s, bins=bin_edges)
            total += counts
    return total

left_rate  = _bin_group(dn_left_idx)
right_rate = _bin_group(dn_right_idx)
both_rate  = _bin_group(dn_both_idx)

# Forward drive: total DN activity, smoothed and normalised to [0, 1]
total_rate   = left_rate + right_rate + both_rate
window       = max(1, N_DECISIONS // 5)
total_smooth = np.convolve(total_rate, np.ones(window) / window, mode="same")
dn_amp       = total_smooth / (total_smooth.max() + 1e-6)

# Turning bias: normalised left-right asymmetry in [-1, 1]
# Positive  -> more left DNs firing  -> bias turning left
# Negative  -> more right DNs firing -> bias turning right
lr_diff = (left_rate - right_rate) / (left_rate + right_rate + 1e-6)

# Build control signal array: shape (N_DECISIONS, 2) = [left_amp, right_amp]
# left/right amplitude each in [0, 1]; reducing one side turns the fly
control_signals = np.zeros((N_DECISIONS, 2))
for t in range(N_DECISIONS):
    base = MIN_AMP + (1.0 - MIN_AMP) * dn_amp[t]   # forward speed driven by total DN
    bias = lr_diff[t] * 0.4                          # turning component
    control_signals[t, 0] = np.clip(base - bias, 0.0, 1.0)   # left legs
    control_signals[t, 1] = np.clip(base + bias, 0.0, 1.0)   # right legs

print(f"  Total DN amp range : [{dn_amp.min():.3f}, {dn_amp.max():.3f}]")
print(f"  LR bias range      : [{lr_diff.min():.3f}, {lr_diff.max():.3f}]")
print(f"  Control signal     : [{control_signals.min():.3f}, {control_signals.max():.3f}]")

# ── 4. Build flygym simulation ────────────────────────────────────────────────
print("\nBuilding flygym simulation ...")

# Place a food odor source 20 mm ahead of the spawn position
odor_source         = np.array([[20.0, 0.0, 1.5]])
peak_odor_intensity = np.array([[1.0, 0.0]])   # attractive dimension only

arena = OdorArena(
    odor_source=odor_source,
    peak_odor_intensity=peak_odor_intensity,
    diffuse_func=lambda x: x**-2,
    marker_colors=[],   # empty list skips marker geom entirely
)

# HybridTurningController requires contact sensors on these segments
contact_sensor_placements = [
    f"{leg}{seg}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

fly = Fly(
    spawn_pos=(0, 0, 0.2),
    spawn_orientation=(0, 0, 0),
    contact_sensor_placements=contact_sensor_placements,
    enable_olfaction=True,
    enable_adhesion=True,
    draw_adhesion=False,
)

# YawOnlyCamera: follows fly heading.
# "camera_top_right" preset: pos=[0,-8,5], elevated right-side isometric view.
cam = YawOnlyCamera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_top_right",
    targeted_fly_names=fly.name,
    timestamp_text=False,
    play_speed_text=False,
    play_speed=0.25,
    fps=30,
)

sim = HybridTurningController(
    fly=fly,
    cameras=[cam],
    arena=arena,
    timestep=PHYSICS_TIMESTEP,
)

# ── Timeline: WALK-1 | GROOM | WALK-2  =  3.75 s physics  ->  15 s video ──────
# Tile the 1 s brain signal to fill 3.75 s of physics -> 15 s video at play_speed=0.25
VIDEO_DURATION_S  = 15.0
PHYS_DURATION_S   = VIDEO_DURATION_S * 0.25          # 3.75 s
N_DECISIONS_TOTAL = int(PHYS_DURATION_S / DECISION_INTERVAL)  # 150
ctrl_tiled = np.tile(control_signals, (int(np.ceil(N_DECISIONS_TOTAL / N_DECISIONS)), 1))
ctrl_tiled = ctrl_tiled[:N_DECISIONS_TOTAL]

print(f"  Physics timestep : {PHYSICS_TIMESTEP} s")
print(f"  Total decisions  : {N_DECISIONS_TOTAL} ({PHYS_DURATION_S:.2f} s physics "
      f"-> {VIDEO_DURATION_S:.0f} s video at 0.25x)")

# ── 5. Run physics loop ────────────────────────────────────────────────────────
print(f"\nStepping physics ({N_DECISIONS_TOTAL * PHYSICS_STEPS_PER_DECISION:,} total steps) ...")
obs, _ = sim.reset()

for t in range(N_DECISIONS_TOTAL):
    ctrl = ctrl_tiled[t]
    for _ in range(PHYSICS_STEPS_PER_DECISION):
        obs, _, _, _, _ = sim.step(ctrl)
        sim.render()

final_pos = obs["fly"][0, :2]
print(f"  Done.  Fly position: x={final_pos[0]:.2f} mm, y={final_pos[1]:.2f} mm")
dist_to_food = float(np.linalg.norm(final_pos - odor_source[0, :2]))
print(f"  Distance to food   : {dist_to_food:.2f} mm  (started at 20.00 mm)")

# ── 6. Save versioned video ───────────────────────────────────────────────────
sim_dir  = Path(__file__).parent / "simulations"
sim_dir.mkdir(exist_ok=True)
existing = sorted(sim_dir.glob("v*_brain_body_*.mp4"))
next_v   = len(existing) + 1 + 2   # +2 for v1/v2 (wing/walk examples)
out_path = sim_dir / f"v{next_v}_brain_body_flygym_odor.mp4"

print(f"\nSaving {out_path} ...")
cam.save_video(str(out_path), stabilization_time=0.02)

print(f"Done - {out_path}")
print(f"All simulations: {[p.name for p in sorted(sim_dir.glob('v*.mp4'))]}")

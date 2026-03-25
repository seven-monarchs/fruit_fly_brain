"""
Brain-body simulation: Drosophila brain model drives MuJoCo walking.

Pipeline
--------
1. Run one trial of the LIF connectome model (Brian2) with Poisson
   stimulation on a small set of "sensory" neurons.
2. Bin the spike trains into time windows matching physics steps.
3. Map the most-active neurons' rates to the 78 actuators of the fly model.
4. Step the raw MuJoCo physics (floor.xml) with those control signals.
5. Save the rendered frames as brain_body_simulation.mp4.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from pathlib import Path

# flybody/dm_control must be imported BEFORE load_dotenv() so the renderer
# auto-detects the correct backend.  The .env file sets MUJOCO_GL=egl which
# is Linux-only and causes an ImportError on Windows if loaded first.
import flybody
from dm_control import mjcf
from flybody.utils import any_substr_in_str

from dotenv import load_dotenv
load_dotenv()

# ── paths ─────────────────────────────────────────────────────────────────────
BRAIN_DIR    = Path(__file__).parent / "brain_model"
PATH_COMP    = BRAIN_DIR / "Completeness_783.csv"
PATH_CON     = BRAIN_DIR / "Connectivity_783.parquet"
FLYBODY_PATH = os.path.dirname(flybody.__file__)
XML_FLOOR    = os.path.join(FLYBODY_PATH, "fruitfly/assets/floor.xml")

sys.path.insert(0, str(BRAIN_DIR))

# Brian2 imports
from brian2 import (
    NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network,
    mV, ms, Hz, second, start_scope, prefs,
)
from model import create_model, poi, default_params

# ── simulation knobs ──────────────────────────────────────────────────────────
BRAIN_DURATION_MS    = 500   # ms of neural simulation
N_PHYSICS_FRAMES     = 200   # rendered frames
PHYS_STEPS_PER_FRAME = 50    # physics substeps per frame (= 5 ms per frame)
N_SENSORY            = 20    # neurons driven by Poisson input
STIM_RATE_HZ         = 150   # Poisson stimulation rate (Hz)
CAMERA               = 6     # 6 = 'hero' (nice perspective view of the fly)
FRAME_W, FRAME_H     = 640, 480

# Actuator indices to drive with neural signals (legs + mouthparts only).
# Wings (14-19) are excluded — they flip the body when driven arbitrarily.
ACTIVE_ACTUATORS = list(range(3, 8)) + list(range(22, 70))
# 3-7:   rostrum, haustellum, labrum (mouth)
# 22-69: coxa / femur / tibia / tarsus, all 3 leg pairs L+R

# Adhesion claw indices — kept at maximum grip at all times to prevent slipping.
ADHESION_ACTUATORS = list(range(70, 78))   # labrum + T1/T2/T3 claw pads

# Neural signals are blended with the neutral pose at this weight (0=neutral, 1=full range).
# Reducing this damps the joint excursions and stops the fly toppling.
NEURAL_BLEND = 0.25

# ── 1. Load connectome index ──────────────────────────────────────────────────
print(f"Loading connectome ({PATH_COMP.name}) ...")
df_comp = pd.read_csv(PATH_COMP, index_col=0)
n_neurons = len(df_comp)
print(f"  {n_neurons:,} neurons in model")

# ── 1b. Warm up MuJoCo renderer BEFORE Brian2 touches anything ────────────────
print("Initialising physics (flat floor) ...")
physics = mjcf.Physics.from_xml_path(XML_FLOOR)
n_ctrl  = physics.model.nu          # 78 actuators
print(f"  {n_ctrl} actuators, {physics.model.ncam} cameras")

# Put the fly in a natural standing pose (same as main.py).
with physics.reset_context():
    for i in range(physics.model.njnt):
        name = physics.model.id2name(i, "joint")
        if any_substr_in_str(["coxa", "femur", "tibia", "tarsus"], name):
            physics.named.data.qpos[name] = physics.named.model.qpos_spring[name]

physics.render(camera_id=CAMERA, width=FRAME_W, height=FRAME_H)  # warm up OpenGL
print("  Physics environment ready")

# ── 2. Run the LIF brain model (single trial) ─────────────────────────────────
print(f"\nRunning LIF brain model ({BRAIN_DURATION_MS} ms, 1 trial) ...")

start_scope()
prefs.codegen.target = "numpy"   # avoid Cython (no Visual Studio on Windows)

params = {**default_params,
          "t_run" : BRAIN_DURATION_MS * ms,
          "n_run" : 1,
          "r_poi" : STIM_RATE_HZ * Hz}

# Build FlyWire ID → Brian index lookup (used for both input and output).
flyid2i = {flyid: i for i, flyid in enumerate(df_comp.index)}

# ── Sensory input: ascending neurons (VNC → brain) ────────────────────────────
# Ascending neurons carry proprioceptive / motor feedback from the legs up to
# the brain. Connectivity analysis shows they make 646 direct synapses onto
# locomotion-related DNs (MDN, DNa01/02, DNg02, DNp09/10/13/42), reaching 28
# of them — far more than any other sensory population (JO: 3, mechan.: 3).
# Stimulating them mimics the brain receiving "the legs are moving" signal.
df_ann = pd.read_csv(BRAIN_DIR / "flywire_annotations.tsv",
                     sep="\t", low_memory=False)
ascending_ids = df_ann[df_ann["super_class"] == "ascending"]["root_id"].tolist()
sensory_idx = [flyid2i[fid] for fid in ascending_ids if fid in flyid2i]
print(f"  Stimulating {len(sensory_idx)} ascending neurons (leg proprioception -> brain)")

# ── Motor readout: descending neurons (DNs) ───────────────────────────────────
# 1,299 DNs identified from FlyWire v783 annotations (Schlegel et al. 2024).
# These are the only neurons that physically connect the brain to the motor
# system in the ventral nerve cord — the true brain→body interface.
#
# Biologically-grounded actuator mapping:
#   DNa01 / DNa02  →  front legs (T1)          [steering / walking initiation]
#   MDN            →  all legs (backward walk)
#   DNg02          →  middle + hind legs (T2/T3)
#   DNp09 / DNp10  →  hind legs (T3)
#   all other DNs  →  distributed via fixed random projection
dn_df = pd.read_csv(BRAIN_DIR / "descending_neurons.csv")

# Map from cell_type to actuator group (leg indices in ACTIVE_ACTUATORS list).
# ACTIVE_ACTUATORS = mouth(0-4) + T1L(5-11) + T1R(12-18) + T2L(19-25) +
#                   T2R(26-32) + T3L(33-39) + T3R(40-46)
DN_TO_LEGS = {
    "DNa01": list(range(5, 19)),           # T1 left + right
    "DNa02": list(range(5, 19)),           # T1 left + right
    "MDN":   list(range(5, 47)),           # all legs
    "DNg02_a": list(range(19, 47)),        # T2 + T3
    "DNg02_b": list(range(19, 47)),        # T2 + T3
    "DNp09": list(range(33, 47)),          # T3 left + right
    "DNp10": list(range(33, 47)),          # T3 left + right
    "DNp13": list(range(19, 47)),          # T2 + T3
    "DNp42": list(range(5, 47)),           # all legs
}

dn_brian_idx = []
dn_leg_targets = []   # which actuator slots each DN influences
for _, row in dn_df.iterrows():
    fid = row["root_id"]
    if fid in flyid2i:
        bidx = flyid2i[fid]
        dn_brian_idx.append(bidx)
        ctype = str(row["cell_type"])
        dn_leg_targets.append(DN_TO_LEGS.get(ctype, None))  # None = generic

n_dn = len(dn_brian_idx)
print(f"  DN motor readout: {n_dn} neurons ({len(dn_df)} total DNs)")

neu, syn, spk_mon = create_model(PATH_COMP, PATH_CON, params)
poi_inp, neu = poi(neu, sensory_idx, [], params)

net = Network(neu, syn, spk_mon, *poi_inp)
net.run(duration=params["t_run"])

raw_trains = spk_mon.spike_trains()
n_spiking  = sum(1 for v in raw_trains.values() if len(v) > 0)
print(f"  {n_spiking:,} neurons fired at least once")

# ── 3. Bin descending neuron spike trains ─────────────────────────────────────
duration_s = BRAIN_DURATION_MS / 1000.0
bin_edges  = np.linspace(0.0, duration_s, N_PHYSICS_FRAMES + 1)

dn_rates  = np.zeros((N_PHYSICS_FRAMES, n_dn), dtype=np.float32)
n_dn_fired = 0
for col, brian_idx in enumerate(dn_brian_idx):
    if brian_idx in raw_trains and len(raw_trains[brian_idx]) > 0:
        times_s = np.array(raw_trains[brian_idx] / second)
        counts, _ = np.histogram(times_s, bins=bin_edges)
        dn_rates[:, col] = counts
        n_dn_fired += 1

n_total_fired = sum(1 for v in raw_trains.values() if len(v) > 0)
print(f"  Total neurons fired : {n_total_fired}")
print(f"  DNs fired           : {n_dn_fired} / {n_dn}")

# ── 4. Build control sequence: tripod gait scaled by DN amplitude ─────────────
#
# Fly walking uses an alternating tripod gait:
#   Tripod A (swing together): T1_right, T2_left,  T3_right
#   Tripod B (swing together): T1_left,  T2_right, T3_left
#
# The descending neuron total firing rate becomes the "go" signal that sets
# the gait speed and amplitude — more DN activity = bigger, faster steps.

n_active     = len(ACTIVE_ACTUATORS)
ctrl_min_all = physics.model.actuator_ctrlrange[:, 0]
ctrl_max_all = physics.model.actuator_ctrlrange[:, 1]
ctrl_mid     = (ctrl_min_all + ctrl_max_all) / 2.0

# DN amplitude: total spikes across all DNs per frame, smoothed + normalised.
dn_total  = dn_rates.sum(axis=1).astype(float)
window    = max(1, N_PHYSICS_FRAMES // 20)
dn_smooth = np.convolve(dn_total, np.ones(window) / window, mode="same")
dn_amp    = dn_smooth / (dn_smooth.max() + 1e-6)   # [0, 1]

print(f"  DNs fired           : {n_dn_fired} / {n_dn}")
print(f"  DN amplitude range  : [{dn_amp.min():.3f}, {dn_amp.max():.3f}]")

# Gait phase — 4 complete step cycles over the simulation.
GAIT_CYCLES = 4
phase = np.linspace(0, 2 * np.pi * GAIT_CYCLES, N_PHYSICS_FRAMES)

TRIPOD_A = {"T1R", "T2L", "T3R"}   # in-phase group
# Leg → start index within ACTIVE_ACTUATORS (5 mouth actuators come first).
# Each leg has 8 joints: coxa_abduct(0) coxa_twist(1) coxa(2) femur_twist(3)
#                        femur(4) tibia(5) tarsus(6) tarsus2(7)
LEG_BASE = {
    "T1L": 5,  "T1R": 13,
    "T2L": 21, "T2R": 29,
    "T3L": 37, "T3R": 45,
}

ctrl_seq = np.tile(ctrl_mid, (N_PHYSICS_FRAMES, 1))  # start at neutral

for leg, base in LEG_BASE.items():
    lp = phase if leg in TRIPOD_A else phase + np.pi   # antiphase for tripod B

    for joint_off, amplitude_frac, ph_off, clip_positive in [
        # (joint index within leg, fraction of ctrlrange, extra phase, lift-only?)
        (2, 0.30, 0.0,        False),   # coxa:  forward / backward swing
        (4, 0.25, np.pi / 2,  True),    # femur: lift during swing phase only
        (5, 0.15, 0.0,        False),   # tibia: extend / flex with coxa
    ]:
        ai = base + joint_off            # index in ACTIVE_ACTUATORS list
        if ai >= n_active:
            continue
        gi = ACTIVE_ACTUATORS[ai]        # global actuator index (0..77)

        half_range = (ctrl_max_all[gi] - ctrl_min_all[gi]) * amplitude_frac / 2.0
        signal     = np.sin(lp + ph_off)
        if clip_positive:
            signal = np.maximum(signal, 0.0)   # only lift, no active push-down

        ctrl_seq[:, gi] = ctrl_mid[gi] + signal * half_range * dn_amp

# Pin adhesion pads to maximum grip so feet stay on the floor.
ctrl_seq[:, ADHESION_ACTUATORS] = ctrl_max_all[ADHESION_ACTUATORS]

print(f"\nMotor signal stats:")
print(f"  active neurons used : {n_dn_fired} DNs  gait cycles: {GAIT_CYCLES}")
print(f"  ctrl range (legs)   : [{ctrl_seq[:, ACTIVE_ACTUATORS].min():.3f}, "
      f"{ctrl_seq[:, ACTIVE_ACTUATORS].max():.3f}]")

# ── 5. Step physics and render ────────────────────────────────────────────────
print(f"\nStepping physics ({N_PHYSICS_FRAMES} frames × {PHYS_STEPS_PER_FRAME} substeps) ...")

# Reset to standing pose before the simulation loop.
with physics.reset_context():
    for i in range(physics.model.njnt):
        name = physics.model.id2name(i, "joint")
        if any_substr_in_str(["coxa", "femur", "tibia", "tarsus"], name):
            physics.named.data.qpos[name] = physics.named.model.qpos_spring[name]

frames = []
for step in range(N_PHYSICS_FRAMES):
    physics.data.ctrl[:] = ctrl_seq[step]
    for _ in range(PHYS_STEPS_PER_FRAME):
        physics.step()
    frame = physics.render(camera_id=CAMERA, width=FRAME_W, height=FRAME_H)
    frames.append(frame)

print(f"  Rendered {len(frames)} frames")

# ── 6. Save video ─────────────────────────────────────────────────────────────
sim_dir = Path(__file__).parent / "simulations"
sim_dir.mkdir(exist_ok=True)
existing = sorted(sim_dir.glob("v*_brain_body_*.mp4"))
next_v   = len(existing) + 1 + 2   # +2 because v1/v2 are wing/walk examples
out_path = sim_dir / f"v{next_v}_brain_body_tripod_gait.mp4"
print(f"\nSaving {out_path} ...")

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")
im = ax.imshow(frames[0])
ax.set_title("LIF brain model → MuJoCo body (flat floor)", fontsize=10)

def _update(frame):
    im.set_data(frame)
    return [im]

anim = animation.FuncAnimation(fig, _update, frames=frames, interval=50, blit=True)
anim.save(out_path, writer="ffmpeg", fps=20)
plt.close()

print(f"Done - saved to {out_path}")
print(f"All simulations: {[p.name for p in sorted(sim_dir.glob('v*.mp4'))]}")

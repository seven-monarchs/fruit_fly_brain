# Fly Brain-Body Simulation — Technical Documentation

## Overview

This project couples a biologically-accurate spiking neural network model of the *Drosophila melanogaster* (fruit fly) brain to a physics-based body simulation. The brain model drives the body in real time: neural activity in the brain determines how fast and in what direction the fly walks.

**In plain terms:** we run a simulation of 138,639 neurons firing and talking to each other, then use the output of that simulation to steer a 3D physics model of a fly body.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [How to Run](#how-to-run)
3. [Biology Glossary](#biology-glossary)
4. [The Full Pipeline, Step by Step](#the-full-pipeline-step-by-step)
5. [Brain Model Deep Dive](#brain-model-deep-dive)
6. [Body Model Deep Dive](#body-model-deep-dive)
7. [Brain-to-Body Interface](#brain-to-body-interface)
8. [Configuration Parameters](#configuration-parameters)
9. [Data Files](#data-files)
10. [Dependencies](#dependencies)
11. [Known Limitations & Design Decisions](#known-limitations--design-decisions)
12. [Version History](#version-history)

---

## Repository Structure

```
fly_brain_simulation/
├── brain_body_simulation.py   # v1: flybody body, hand-coded tripod gait
├── brain_body_v2.py           # v2: flygym body, CPG gait, OdorArena  ← MAIN
├── main.py                    # Standalone wing animation demo
├── walking_ani.py             # Standalone random-policy walking demo
├── set_token.py               # FlyWire API token setup (connectome access)
├── .env                       # MuJoCo renderer config (MUJOCO_GL=egl)
├── brain_model/
│   ├── model.py               # Brian2 LIF network builder
│   ├── utils.py               # Spike analysis helpers
│   ├── Completeness_783.csv   # Neuron index: 138,639 neurons (FlyWire v783)
│   ├── Connectivity_783.parquet  # Synapse table: ~50M connections
│   ├── flywire_annotations.tsv   # Cell-type labels for all neurons
│   └── descending_neurons.csv    # 1,299 motor output neurons with side labels
└── simulations/
    └── v*.mp4                 # Versioned output videos
```

---

## How to Run

```bash
# Activate the virtual environment
# Windows:
wenv310\Scripts\activate

# Run the main simulation (generates a new versioned .mp4 in simulations/)
python brain_body_v2.py

# Run the older v1 simulation (flybody, tripod gait)
python brain_body_simulation.py
```

> **Important on Windows:** `flygym` / `dm_control` must be imported **before** `load_dotenv()` is called. The `.env` file sets `MUJOCO_GL=egl` which is a Linux-only GPU renderer. If it is loaded first, the import fails. The import order in the scripts is intentional — do not reorder it.

---

## Biology Glossary

Understanding this project requires a few neuroscience and entomology concepts. All terms are defined here in developer-friendly language.

### Drosophila melanogaster
The common fruit fly. A model organism in biology — its nervous system is small enough to map completely (~140,000 neurons) but complex enough to produce real behaviors like walking, flying, grooming, and odor-guided navigation. Think of it as the "Hello World" of neuroscience.

### Connectome
A complete map of every neuron and every synapse (connection between neurons) in a nervous system. The *Drosophila* connectome was assembled from electron microscopy images by the FlyWire project. Version 783 (used here) contains **138,639 neurons** and roughly **50 million synapses**. This is the equivalent of a complete wiring diagram for the brain.

### Neuron
The basic computational unit of the nervous system. Receives electrical inputs, integrates them, and fires an output spike if the integrated input crosses a threshold. Modeled here as a **Leaky Integrate-and-Fire (LIF)** unit (see below).

### Synapse
A connection from one neuron to another. Each synapse has a **weight** (strength) and a **sign** (excitatory = activates the target, inhibitory = suppresses it). In the connectome data, sign is encoded as `Excitatory x Connectivity` — positive values are excitatory, negative are inhibitory.

### Spike / Action Potential
A neuron "fires" when its membrane voltage crosses a threshold, producing a brief electrical pulse called a spike. All information in biological neural networks is encoded as sequences of these discrete events (spike trains). Think of it as a 1-bit digital signal.

### Spike Train
The sequence of timestamps at which a neuron fired during a simulation. In Brian2, `spike_monitor.spike_trains()` returns a dictionary of `{neuron_id: array_of_spike_times}`.

### Leaky Integrate-and-Fire (LIF) Model
A simplified mathematical model of a neuron. The membrane voltage `v` leaks back to a resting potential `v_0` over time (like a capacitor discharging), but integrates incoming spikes through a conductance variable `g`. When `v` crosses the threshold `v_th`, the neuron fires and resets.

The equations (from `brain_model/model.py`):
```
dv/dt = (v_0 - v + g) / t_mbr   [membrane dynamics]
dg/dt = -g / tau                  [synaptic conductance decay]
```

Where:
- `v_0 = -52 mV` — resting potential (voltage when no input)
- `v_th = -45 mV` — threshold (fire when v exceeds this)
- `v_rst = -52 mV` — reset potential (after firing, v drops back here)
- `t_mbr = 20 ms` — membrane time constant (how fast voltage leaks back)
- `tau = 5 ms` — synaptic time constant (how fast a received spike decays)
- `t_rfc = 2.2 ms` — refractory period (minimum time between spikes; neuron is "deaf" to input during this window)

### Poisson Input
A way to model external stimulation. A **Poisson process** generates random spike events at a fixed average rate (e.g., 150 Hz = 150 spikes/second on average). Each incoming spike adds voltage to the target neuron's `g` variable. This simulates a neuron receiving noisy but sustained input from somewhere outside the modeled network.

### Ascending Neurons
Neurons that carry signals **from the body up to the brain**. In the fly, they physically run from the **Ventral Nerve Cord (VNC)** — the fly's equivalent of a spinal cord — up into the brain. They carry proprioceptive information (where the legs are, how fast they're moving). We stimulate 1,736 of these as our sensory input, simulating "the legs are moving."

> **Why ascending neurons?** Connectivity analysis showed they make 646 direct synapses onto locomotion-related descending neurons — far more than any other sensory type. Stimulating them most efficiently activates the locomotion circuitry.

### Descending Neurons (DNs)
Neurons that carry signals **from the brain down to the body**. They are the only physical pathway from the brain to the motor system. In the fly there are ~1,299 annotated DNs. They are the brain's output — the "motor commands." We record their activity and use it to drive the physics body.

Notable DNs used for biologically-grounded mapping:
- **DNa01 / DNa02** — front leg control, walking initiation and steering
- **MDN (Moon-walking DN)** — drives backward walking
- **DNg02** — middle and hind leg coordination
- **DNp09 / DNp10** — hind leg control

### Ventral Nerve Cord (VNC)
The fly's "spinal cord." Sits below the brain and contains the actual **Central Pattern Generator (CPG)** circuits that produce rhythmic leg movements. This is the missing piece in our simulation — we bypass it by using flygym's built-in CPGNetwork. Bridging the real connectome's VNC is a major open research problem.

### Central Pattern Generator (CPG)
A neural circuit that produces rhythmic output without rhythmic input. CPGs exist in all animals and are responsible for walking, swimming, breathing, etc. They do not need sensory feedback to run — they oscillate on their own. In flygym, the CPG is implemented as 6 coupled oscillators (one per leg) with phase relationships that produce a tripod gait.

### Tripod Gait
The walking pattern used by flies and many other insects. The six legs are divided into two groups of three:
- **Tripod A (swing phase):** front-right, middle-left, hind-right lift off simultaneously
- **Tripod B (stance phase):** front-left, middle-right, hind-left stay on the ground

The two tripods alternate — while A is in the air, B supports the body, and vice versa. This always maintains at least 3 feet on the ground, providing stable support. It is the insect equivalent of diagonal trotting in quadrupeds.

### Proprioception
The sense of where your own body parts are in space. In the fly context: the legs send signals back to the brain reporting their position and force. Ascending neurons carry this information. We model it by stimulating those ascending neurons with Poisson inputs.

### Olfaction / Odor Taxis
The sense of smell. **Odor taxis** is navigation guided by smell gradients — moving toward attractive smells, away from aversive ones. The fly detects odor through antennae and maxillary palps. In the `OdorArena`, odor concentration follows an inverse-square law (`C ∝ 1/r²`). The fly's olfactory sensors read the local concentration, and the resulting left/right asymmetry drives turning. In our simulation, the brain drives locomotion independently — we do not use the olfactory readings to feed back into the brain.

### Membrane Potential
The electrical voltage across a neuron's cell membrane. Measured in millivolts (mV). At rest: ~-52 mV. Spikes when: ~-45 mV. After spiking, resets back to: ~-52 mV.

---

## The Full Pipeline, Step by Step

```
[FlyWire Connectome Data]
        │
        ▼
[1. Build Brian2 LIF Network]
   138,639 neurons, ~50M synapses
        │
        ▼
[2. Stimulate Ascending Neurons]
   1,736 neurons, Poisson @ 150 Hz
   (simulates "legs are moving" signal from body to brain)
        │
        ▼
[3. Run Brain Simulation — 1000 ms]
   ~15,000 neurons fire at least once
        │
        ▼
[4. Record Descending Neuron Spikes]
   645 left DNs + 646 right DNs + 8 bilateral DNs
        │
        ▼
[5. Bin Spike Trains — 25 ms windows]
   left_rate[40], right_rate[40], both_rate[40]
        │
        ▼
[6. Compute Control Signal]
   total_amp → forward speed
   left vs right asymmetry → turning bias
   → control_signals[40, 2]
        │
        ▼
[7. Tile Signal to 3.75 s]
   ctrl_tiled[150, 2] (brain signal loops ~3.75x)
        │
        ▼
[8. Step flygym Physics — 37,500 steps]
   HybridTurningController + CPGNetwork + OdorArena
        │
        ▼
[9. Capture & Save Video]
   YawOnlyCamera → vN_brain_body_flygym_odor.mp4
```

---

## Brain Model Deep Dive

### Source
Based on [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model), which implements the network from:
> Shiu et al. (2023). A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor processing. *PLOS Computational Biology.*

### Neuron Model (`brain_model/model.py`)

```python
eqs = '''
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
'''
```

Parameters (from `default_params`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `v_0` | -52 mV | Resting potential |
| `v_rst` | -52 mV | Post-spike reset voltage |
| `v_th` | -45 mV | Spike threshold |
| `t_mbr` | 20 ms | Membrane time constant |
| `tau` | 5 ms | Synaptic conductance decay |
| `t_rfc` | 2.2 ms | Refractory period |
| `t_dly` | 1.8 ms | Synaptic transmission delay |
| `w_syn` | 0.275 mV | Weight per synapse |
| `r_poi` | 150 Hz | Poisson stimulation rate |
| `f_poi` | 250 | Poisson weight scaling factor |

### Connectivity (`Connectivity_783.parquet`)

Each row is a synapse:

| Column | Description |
|--------|-------------|
| `Presynaptic_Index` | Brian2 neuron index of the sender |
| `Postsynaptic_Index` | Brian2 neuron index of the receiver |
| `Excitatory x Connectivity` | Signed synapse strength (positive = excitatory, negative = inhibitory) |

The synaptic weight in Brian2 is: `w = (Excitatory x Connectivity) × w_syn`

### Why numpy backend?
```python
prefs.codegen.target = "numpy"
```
Brian2 by default compiles to C++ via Cython for speed. This requires a C++ compiler (Visual Studio on Windows). To avoid that dependency, we force the pure-Python numpy backend. It is ~10x slower but requires no compiler.

### Sensory Input Choice

We stimulate **ascending neurons** rather than sensory neurons (olfactory, visual, etc.) because:
1. They have 646 direct synapses onto locomotion DNs, reaching 28 of them
2. This is far more than any other sensory type (Johnston's Organ neurons: 41 synapses, only 3 DNs)
3. Biologically, this simulates proprioceptive feedback from walking legs — a self-sustaining loop

---

## Body Model Deep Dive

### Source
[NeuroMechFly v2](https://neuromechfly.org/) (flygym library), from:
> Lobato-Rios et al. (2023). NeuroMechFly 2.0, a framework for simulating embodied sensorimotor control in adult Drosophila. *Nature Methods.*

### Physics Engine
[MuJoCo](https://mujoco.org/) via [dm_control](https://github.com/google-deepmind/dm_control). MuJoCo stands for **MuJoCo** (Multi-Joint dynamics with Contact). It is a high-performance physics simulator developed by DeepMind, standard in robotics and biomechanics research.

### Fly Model Specifications

| Property | Value |
|----------|-------|
| Degrees of freedom | 87 joints |
| Leg joints per leg | 7 (Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1) |
| Total leg joints | 42 (6 legs × 7 DOF) |
| Adhesion pads | Enabled (prevents feet from slipping) |
| Olfactory sensors | Enabled (antennae + maxillary palps, 4 sensors total) |
| Physics timestep | 0.1 ms (1e-4 s) |
| Gravity | -9810 mm/s² (standard gravity, model is in mm) |

### Joint Naming Convention

Legs are named by position and side:
- `LF` = Left Front, `LM` = Left Middle, `LH` = Left Hind
- `RF` = Right Front, `RM` = Right Middle, `RH` = Right Hind

Example joint name: `LFFemur` = Left Front leg, Femur joint.

The full leg DOF order (42 values in the action vector):
```
LF: LFCoxa, LFCoxa_roll, LFCoxa_yaw, LFFemur, LFFemur_roll, LFTibia, LFTarsus1
LM: LMCoxa, ...
LH: LHCoxa, ...
RF: RFCoxa, ...
RM: RMCoxa, ...
RH: RHCoxa, ...
```

### HybridTurningController

The main controller class. Inherits from `SingleFlySimulation` (a Gymnasium-compatible environment). Takes a 2D control signal each step:

```python
obs, reward, terminated, truncated, info = sim.step(control_signal)
# control_signal: np.array([left_amplitude, right_amplitude])
# Both values in [0, 1]
```

Internally it:
1. Sets CPG intrinsic amplitudes from `control_signal`
2. Steps 6 coupled oscillators (one per leg)
3. Applies retraction rule (lifts legs that get stuck in gaps)
4. Applies stumbling rule (corrects legs that hit obstacles)
5. Calls `physics.step()` (1 MuJoCo physics step = 0.1 ms)

**Turning mechanism:** Reducing one side's amplitude slows those legs → the fly curves toward that side.
- `[1.0, 1.0]` → straight ahead at full speed
- `[0.5, 1.0]` → turn left (left legs slower)
- `[1.0, 0.5]` → turn right (right legs slower)
- `[0.3, 0.3]` → slow walk (both sides at minimum)

### CPGNetwork

6 oscillators, one per leg, with **tripod phase biases**. The phase relationship matrix ensures Tripod A (LF, RM, RH) and Tripod B (RF, LM, LH) oscillate in antiphase (π radians apart). Default frequency: 12 Hz (12 complete step cycles per second).

### OdorArena

A flat ground plane with odor point sources. Odor concentration at position `r` from a source: `C(r) = peak_intensity / r²` (inverse-square diffusion). The fly's 4 olfactory sensors (2 antennae + 2 maxillary palps) each read the local concentration. Left/right sensor asymmetry can drive turning — this is the biological basis of odor navigation.

In our simulation: the odor source exists in the arena but the brain model does not yet read or respond to it. It is infrastructure for future integration.

---

## Brain-to-Body Interface

This is the core of the project. The biological reality is that descending neurons (DNs) are the **only wires** connecting the brain to the motor system. We read their activity and translate it into a 2D locomotion command.

### Step 1: Bin spike trains

The 1000 ms brain simulation is divided into 40 bins of 25 ms each. For each bin, we count how many spikes were fired by each DN group:

```python
bin_edges = np.linspace(0.0, 1.0, 41)   # 40 bins
left_rate[t]  = sum of spikes in bin t from left-side DNs
right_rate[t] = sum of spikes in bin t from right-side DNs
both_rate[t]  = sum of spikes in bin t from bilateral DNs
```

### Step 2: Forward drive (amplitude)

Total DN activity → how fast the fly walks:
```python
total_rate = left_rate + right_rate + both_rate
# Smooth with moving average (window = 8 bins = 200 ms)
total_smooth = convolve(total_rate, uniform_window)
# Normalize to [0, 1]
dn_amp = total_smooth / total_smooth.max()
```

### Step 3: Turning bias (asymmetry)

Left vs right DN firing difference → which way the fly turns:
```python
lr_diff = (left_rate - right_rate) / (left_rate + right_rate + ε)
# Range: [-1, 1]
# Positive → more left DNs → fly turns left
# Negative → more right DNs → fly turns right
```

### Step 4: Control signal

```python
base = MIN_AMP + (1 - MIN_AMP) * dn_amp[t]   # [0.3, 1.0]
bias = lr_diff[t] * 0.4                        # ±0.4 max

control_signals[t, 0] = clip(base - bias, 0, 1)  # left legs
control_signals[t, 1] = clip(base + bias, 0, 1)  # right legs
```

`MIN_AMP = 0.3` ensures the fly never completely stops — biologically, a fly receiving proprioceptive input from walking legs should keep walking.

### Step 5: Time tiling

The brain runs for 1 second, but we want 3.75 seconds of physics (= 15 seconds of video at 0.25× playback speed). The 40-decision control signal is tiled (repeated) approximately 3.75× to fill the full simulation duration.

---

## Configuration Parameters

All tunable constants are at the top of `brain_body_v2.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `BRAIN_DURATION_MS` | 1000 | How long to run the neural simulation (ms). Longer = more diverse control patterns, but takes more time. |
| `STIM_RATE_HZ` | 150 | Poisson stimulation rate for ascending neurons (Hz). Higher = more neurons fire = stronger locomotion signal. |
| `DECISION_INTERVAL` | 0.025 | Duration of each control window (seconds). Shorter = more responsive to neural fluctuations. |
| `PHYSICS_TIMESTEP` | 1e-4 | MuJoCo physics step size (seconds). Do not increase above 1e-4 or simulation becomes unstable. |
| `MIN_AMP` | 0.3 | Minimum CPG amplitude (0–1). Prevents the fly from stopping when DN activity is low. |
| `VIDEO_DURATION_S` | 15.0 | Target output video length (seconds) at `play_speed`. |
| `play_speed` | 0.25 | Video playback speed relative to real time. 0.25 = 4× slow motion. Physics runs 0.25× as long as the video. |

---

## Data Files

### `brain_model/Completeness_783.csv`
Neuron index table from FlyWire connectome version 783.

| Column | Description |
|--------|-------------|
| Index (row label) | FlyWire root ID (a 64-bit integer unique to each neuron) |
| Other columns | Completeness metadata (axon/dendrite coverage) |

The row order defines the Brian2 neuron index. Neuron `i` in Brian2 corresponds to row `i` in this CSV.

### `brain_model/Connectivity_783.parquet`
Synapse table. Each row is one synapse (directional connection).

| Column | Description |
|--------|-------------|
| `Presynaptic_Index` | Brian2 index of sending neuron |
| `Postsynaptic_Index` | Brian2 index of receiving neuron |
| `Excitatory x Connectivity` | Signed strength: positive = excitatory, negative = inhibitory |

### `brain_model/flywire_annotations.tsv`
Cell-type annotations for all 139,244 neurons (including neurons outside our 783 subset).

Key columns:
- `root_id` — FlyWire ID (matches the CSV index)
- `super_class` — coarse type: `ascending`, `descending`, `visual`, `olfactory`, `central`, etc.
- `cell_class` — finer classification
- `cell_type` — specific named type (e.g., `MDN`, `DNa01`, `DNg02_a`)
- `side` — `left`, `right`, or empty (bilateral)

Source: Schlegel et al. (2024), supplementary data from the FlyWire paper.

### `brain_model/descending_neurons.csv`
Filtered subset: 1,299 descending neurons.

| Column | Description |
|--------|-------------|
| `root_id` | FlyWire ID |
| `cell_type` | Named type (e.g., `MDN`, `DNa01`) |
| `top_nt` | Dominant neurotransmitter |
| `side` | `left`, `right`, or empty |

---

## Dependencies

All installed in `wenv310/` (Python 3.10 virtual environment, Windows):

| Package | Purpose |
|---------|---------|
| `brian2` | Spiking neural network simulation |
| `flygym` (v1.2.1) | NeuroMechFly body model + MuJoCo environment |
| `dm_control` | DeepMind MuJoCo Python bindings |
| `mujoco` | Physics engine |
| `numpy` | Array math |
| `pandas` | Connectome data loading |
| `python-dotenv` | Load `.env` for renderer config |
| `imageio` | Video encoding (used by flygym's Camera) |
| `ffmpeg` | Video codec (system dependency, called by imageio) |

---

## Known Limitations & Design Decisions

### 1. VNC is bypassed
The real fly brain connects to the VNC, which contains the CPGs that produce walking. We replaced the VNC with flygym's `CPGNetwork`. The brain only provides a coarse 2D speed/steering signal — it does not generate individual joint commands. Bridging this gap would require a full VNC connectome model.

### 2. Brain signal is tiled
The brain runs for 1 second and produces 40 control bins. For a 15-second video, this signal is repeated ~3.75×. The fly therefore walks with the same neural "rhythm" looped. A proper fix would be to run the brain continuously alongside the physics, with sensory feedback closing the loop.

### 3. No sensory feedback loop
The ascending neurons are stimulated with fixed Poisson noise — they do not actually respond to the simulated leg movements. A closed-loop system would read leg positions from the physics simulation and use them to set ascending neuron activity. This is the key gap between this work and Eon Systems' result.

### 4. Olfaction not connected to brain
The `OdorArena` sends odor readings to the fly's virtual sensors, but this data is not fed back into the Brian2 model. Connecting olfactory neurons (projection neurons in the antennal lobe) to the brain model would enable true odor-guided navigation.

### 5. Stochastic neural output
Brian2 with Poisson input is stochastic — each run produces slightly different spike trains due to the random Poisson process. This is biologically realistic (neurons are noisy) but means the fly will walk slightly differently each run.

### 6. Import order (Windows-specific)
`flygym` → `load_dotenv()` must be respected. See [How to Run](#how-to-run).

---

## Version History

| File | Video(s) | Description |
|------|----------|-------------|
| `main.py` | v1 | Wing animation, no physics stepping |
| `walking_ani.py` | v2 | Random-policy walking with `walk_imitation` env (includes ghost reference trajectory) |
| `brain_body_simulation.py` | v3–v5 | flybody body, hand-coded tripod gait, ascending neuron input, DN motor readout |
| `brain_body_v2.py` | v6+ | flygym NeuroMechFly v2, HybridTurningController + CPGNetwork, OdorArena, YawOnlyCamera |

### `brain_body_v2.py` sub-versions (tracked by video number)

| Video | Key Change |
|-------|-----------|
| v6 | First flygym run, bird's-eye fixed camera |
| v7 | Extended to 15 s, fixed camera |
| v8 | YawOnlyCamera `camera_top_right` — isometric tracking camera |
| v9 | Added grooming phase (front legs raised toward face) |
| v10 | Attempted `camera_right_front` + more aggressive grooming angles → fly flipped |
| v11+ | Reverted to v8 stable config, no grooming |

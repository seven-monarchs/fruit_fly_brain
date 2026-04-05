# Simulation Data

This folder contains the two biological datasets that drive the fly brain-body simulation.
Both are derived from real neuroscience experiments on *Drosophila melanogaster*; the files
are not synthetic or approximated, they are the actual measured wiring of the fly brain and
the actual trained responses of its visual neurons.

---

## 1. FlyWire Connectome v783: `flywire_connectome_v783/`

### What is it?

A complete map of every neuron and every synaptic connection in the adult *Drosophila* brain,
reconstructed by the FlyWire project using electron microscopy (EM). A small piece of fly
brain tissue was sliced into thousands of ultrathin layers (~40 nm each), each layer was
imaged at nanometre resolution, and the resulting image stack was computationally assembled
into a 3D volume. Neurons were then traced by a combination of AI segmentation and human
proofreaders worldwide. Version 783 is the latest public release.

**Scale:** 138,639 neurons, ~50 million synapses.
This is the first near-complete wiring diagram of any brain.

**Paper:**
> Dorkenwald et al. (2023). *Neuronal wiring diagram of an adult brain.*
> Nature. https://doi.org/10.1038/s41586-024-07558-y

**Source (Shiu et al. brain model):** https://github.com/philshiu/Drosophila_brain_model

---

### Files

#### `Completeness_783.csv` (~3.3 MB)
**What it contains:** One row per neuron, 138,639 rows total.

| Column | Content |
|--------|---------|
| `root_id` | FlyWire neuron ID (unique 64-bit integer) |
| `flow` | Broad functional class (`sensory`, `ascending`, `descending`, `motor`, `intrinsic`) |
| `super_class` | Finer classification (`visual`, `olfactory`, `mechanosensory`, etc.) |
| `cell_class` | Anatomical class (e.g. `LHON`, `MBON`, `DN`) |
| `cell_type` | Specific cell type label where known |
| `nt_type` | Neurotransmitter (glutamate, GABA, acetylcholine, etc.) |

**How it is used in this simulation:**
This is the master index. Its **row order defines the Brian2 neuron index**: neuron at
row 0 is Brian2 neuron 0, row 1 is neuron 1, and so on. Every other file refers to
neurons by this row index. We also use the `flow` and `super_class` columns to identify
subpopulations: ascending sensory neurons (1,736), olfactory neurons (2,279), SEZ/feeding
neurons (408), and descending neurons (1,299) that drive left/right motor asymmetry.

---

#### `Connectivity_783.parquet` (~96 MB)
**What it contains:** Every synapse between every pair of connected neurons, ~50 million rows.

| Column | Content |
|--------|---------|
| `Presynaptic_Index` | Row index in Completeness_783.csv of the sending neuron |
| `Postsynaptic_Index` | Row index of the receiving neuron |
| `Excitatory x Connectivity` | Signed synapse weight (+excitatory, −inhibitory) |

The weight encodes both the number of synaptic contacts and the neurotransmitter sign.
A positive value means the pre-synaptic neuron excites the post-synaptic neuron; negative
means inhibition.

**How it is used:**
This table is loaded at startup and used to build the Brian2 `Synapses` object; the full
50-million synapse connectivity matrix. Each row becomes one Brian2 synapse. Loading this
file takes ~30 seconds and is the most memory-intensive step of initialization (~2 GB RAM).
We use Parquet format (columnar binary) rather than CSV because it loads ~10x faster.

---

#### `flywire_annotations.tsv` (~31 MB)
**What it contains:** Additional annotations per neuron from the FlyWire community, one
row per neuron, matched by `root_id`.

| Column | Content |
|--------|---------|
| `root_id` | FlyWire neuron ID (links to Completeness_783.csv) |
| `super_class`, `cell_class`, `cell_type` | Community-curated type labels |
| `side` | Hemisphere: `left`, `right`, or `center` |
| `soma_x`, `soma_y`, `soma_z` | 3D coordinates of the neuron's cell body (voxels, 4 nm resolution) |
| `pos_x`, `pos_y` | Approximate position for neurons without soma coordinates |

**How it is used:**
Soma coordinates (`soma_x`, `soma_y`) are used to position each neuron in the brain
visualisation panel. The frontal view of the brain in the output video is built by
projecting all 138,639 neurons onto a 2D plane using these coordinates. Neurons without
soma coordinates (olfactory and SEZ neurons) fall back to `pos_x`/`pos_y`. The `side`
column separates left descending neurons from right descending neurons for the motor
asymmetry signal.

---

#### `descending_neurons.csv` (~57 KB)
**What it contains:** A curated list of 1,299 descending neurons (DNs): neurons that
carry commands from the brain down to the ventral nerve cord (fly equivalent of the
spinal cord) to control locomotion.

| Column | Content |
|--------|---------|
| `root_id` | FlyWire neuron ID |
| `side` | `left` or `right` |
| `cell_type` | DN subtype (e.g. DNa01, DNa02, ...) |
| `brian2_index` | Pre-computed row index in the main neuron table |

**How it is used:**
At each 25 ms simulation step, we count how many left DNs spiked vs right DNs spiked.
The difference (`left_count - right_count`) produces a small motor bias (scaled by 0.15)
that adds biologically grounded asymmetry to the fly's steering on top of the odor gradient.
This is the only part of the 138,639-neuron connectome that directly drives body movement.

---

### How to place the files

The simulation reads these files from `brain_model/` at the project root:

```
fly_brain_simulation/
  brain_model/
    Completeness_783.csv        <- place here
    Connectivity_783.parquet    <- place here
    flywire_annotations.tsv     <- place here
    descending_neurons.csv      <- place here
```

No installation step; the Python code reads them directly with `pandas` and `pyarrow`.

---

## 2. flyvis Pretrained Weights: `flyvis_pretrained_weights/`

### What is it?

Trained synaptic weights for a **connectome-constrained neural network** of the fly visual
system, published by Lappalainen et al. (2024) in *Nature*. The architecture is not a
generic deep network; it was designed to match the exact wiring of the fly optic lobe,
with 65 cell types arranged in a hexagonal retinotopic map that mirrors the anatomy of the
compound eye.

The network was trained to reproduce the responses of real fly neurons measured with calcium
imaging and electrophysiology, using the connectome structure as a hard architectural
constraint. The result is a model that both matches the circuit diagram and predicts
biological neural activity.

**Cell types (65 total):**
- **Photoreceptors:** R1-R6, R7, R8: input from 721 ommatidia per eye
- **Lamina:** L1, L2, L3, L4, L5: first processing layer
- **Medulla:** Mi1, Mi4, Mi9, Tm1, Tm2, Tm3, Tm4, Tm9, ...: feature extraction
- **Direction-selective:** T4a/b/c/d, T5a/b/c/d: motion detectors (ON and OFF pathways)

**Paper:**
> Lappalainen et al. (2024). *Connectome-constrained networks predict neural activity across the fly visual system.*
> Nature. https://doi.org/10.1038/s41586-024-07939-3

**Code repository:** https://github.com/TuragaLab/flyvis

---

### Folder structure

```
flyvis_pretrained_weights/
  0000/                     <- ensemble index (one trained model ensemble)
    000/                    <- individual model within the ensemble
      best_chkpt/           <- weights at the training epoch with best validation loss
      chkpts/               <- all saved training checkpoints
      validation/           <- validation metrics
      validation_loss.h5    <- loss curve
```

The simulation loads `0000/000`: the first model of the first ensemble:

```python
nv      = NetworkView(flyvis.results_dir / 'flow/0000/000')
network = nv.init_network(checkpoint='best')
```

---

### What the weights encode

Each weight defines the **synaptic strength between two cell types** at a given hexagonal
offset in the retinotopic map. Because the architecture is constrained by the connectome,
the weight matrix is sparse; connections that don't exist in the real fly optic lobe are
structurally absent and never trained.

The trained network produces direction-selective responses in T4/T5 that match real fly
recordings:
- **T4** responds to bright edges moving in a preferred direction (ON pathway)
- **T5** responds to dark edges expanding in their receptive field (OFF pathway)

A dark obstacle approaching the fly produces an expanding dark region in the visual field,
exactly the stimulus T5 is tuned to detect.

---

### How it is used in this simulation

At each 25 ms simulation step:

1. `obs["vision"]` from MuJoCo gives raw ommatidium luminance: shape `(2, 721, 2)`
   (2 eyes x 721 ommatidia x 2 photoreceptor channels: yellow/pale)
2. `RetinaMapper.flygym_to_flyvis()` reorders the 721 ommatidia from flygym's convention
   to flyvis's hexagonal lattice convention (same data, different index ordering)
3. The remapped frame is fed to the network via a single stateful `forward()` call (~17ms)
4. The network carries its recurrent synaptic state from the previous step; no reset between
   steps, which is both faster and more biologically correct
5. T5a and T5b activity is extracted for both eyes:
   - Left eye T5 > Right eye T5 -> dark motion on left -> steer right
   - Right eye T5 > Left eye T5 -> dark motion on right -> steer left
6. The T5 asymmetry becomes a turn bias (clamped to +-2.0), mixed with the odor gradient

This is the biological avoidance pathway: compound eye -> lamina -> medulla ->
T5 direction-selective cells -> turn command. In the real fly this runs through lobula
plate tangential cells (LPTCs) which integrate T5 signals across the visual field.

---

### How to install

The weights must be placed where the `flyvis` package expects them:

```
wenv310\Lib\site-packages\flyvis\data\results\flow\0000\
```

**Option A: copy from this folder** (already downloaded here):
Copy `flyvis_pretrained_weights/0000/` into the path above.

**Option B: download fresh** using the flyvis CLI:
```bash
wenv310\Scripts\flyvis download-pretrained --skip_large_files
```

The `--skip_large_files` flag skips UMAP embeddings and clustering analysis files
(several GB) and downloads only the checkpoint weights needed for inference (~6 MB).

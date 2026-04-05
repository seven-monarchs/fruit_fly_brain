"""
generate_plots.py  --  NeuroFly simulation analysis plots

Reads the HDF5 data file produced by fly_brain_body_simulation.py and generates
one PNG per plot per language in  plots/vN/EN/  and  plots/vN/FR/

Usage:
    python generate_plots.py simulations/vN_data.h5
    python generate_plots.py          # auto-picks the latest vN_data.h5

Plots generated (x2, EN + FR):
    01_circuit_timeline.png         mean firing rate per circuit over 10 s + events
    02_raster_circuits.png          spike raster for DNs, SEZ, ascending, olfactory
    03_dn_turning_coupling.png      DN L/R asymmetry vs motor control amplitude
    04_brain_body_coupling.png      proprioceptive rate + DN vs turn direction
    05_population_heatmap.png       binned spike density per circuit as 2-D heatmap
    06_firing_rate_distribution.png mean firing rate histogram per circuit
    07_odor_olfactory_response.png  odor sensor intensity vs olfactory activity
    08_visual_lamina_response.png   compound-eye luminance vs LA>ME lamina firing rate
    09_trajectory.png               fly XY path with walls, heading arrows, food
    10_looming_reflex.png           per-eye looming signal + reflex bias over time
    11_odor_asymmetry.png           left vs right antenna odor + steering bias
    12_odor_field_trajectory.png    trajectory overlaid on Dijkstra odor gradient heatmap
    13_odor_movement_correlation.png odor experienced + asymmetry vs turn signal + scatter correlation
"""

import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
#  TRANSLATIONS
# =============================================================================
LANGS = {
    "EN": {
        "time_s":            "Simulation time (s)",
        "dist_food":         "dist to food (mm)",
        "mean_rate_hz":      "Mean rate (Hz)",
        "neuron_count":      "Neuron count",
        "ctrl_amplitude":    "Control amplitude",
        "asymmetry":         "Asymmetry",
        "brain_to_body":     "[brain -> body]",
        "body_to_brain":     "[body -> brain]",
        "odor_norm":         "Odor intensity (normalised)",
        "olf_rate":          "Olfactory mean rate (Hz)",
        "asc_rate_label":    "Ascending rate (Hz)\n[body -> brain]",
        "asym_label":        "Asymmetry\n[brain -> body]",
        "spike_count":       "Spike count / step",
        "dn_lr_diff":        "DN lr_diff",
        "mean_rate_label":   "Mean rate (Hz)",
        "dn_left_leg":       "DN left (inverted)",

        # circuit labels
        "c_asc":   "Ascending / proprio",
        "c_olf":   "Olfactory (sample)",
        "c_sez":   "SEZ / feeding",
        "c_dnl":   "DN left",
        "c_dnr":   "DN right",
        "c_olf_s": "Olfactory\n(sample)",
        "c_dn":    "DN (L+R)",

        # legend entries
        "leg_dn_left":   "DN left",
        "leg_dn_right":  "DN right (inverted)",
        "leg_ctrl_l":    "ctrl left",
        "leg_ctrl_r":    "ctrl right",
        "leg_dn_brain":  "DN lr_diff (brain)",
        "leg_ctrl_body": "ctrl_left - ctrl_right (body)",
        "leg_odor":      "Odor intensity (norm.)",
        "leg_olf":       "Olfactory mean rate (Hz)",

        # titles
        "t01": "Circuit activation timeline",
        "t02": "Spike raster (key circuits)",
        "t03": "Descending neuron activity vs motor output",
        "t04": "Brain-body closed loop coupling",
        "t05": "Population firing rate heatmap (100ms bins)",
        "t06": "Mean firing rate distribution per circuit",
        "t07": "Odor gradient vs olfactory circuit activity",
        "t08": "Compound-eye luminance vs LA>ME lamina neuron firing rate",
        "t_cbar": "Mean rate (Hz)",

        "leg_vis_left":  "Left eye -> left lamina (Hz)",
        "leg_vis_right": "Right eye -> right lamina (Hz)",
        "vis_rate_hz":   "Lamina firing rate (Hz)",
        "luminance":     "Luminance (norm.)",

        "t09": "Fly trajectory with wall layout",
        "t10": "Looming reflex: per-eye signal and steering bias",
        "t11": "Left vs right antenna odor and steering",
        "t12": "Fly trajectory overlaid on channeled odor gradient",
        "x_mm": "X position (mm)", "y_mm": "Y position (mm)",
        "leg_path":      "Fly path",
        "leg_landmark":  "Visual landmark",
        "leg_food":      "Food",
        "leg_spawn":     "Spawn",
        "leg_wall":      "Wall",
        "leg_tunnel":    "Tunnel corridor",
        "leg_loom_l":    "Left-eye loom signal",
        "leg_loom_r":    "Right-eye loom signal",
        "leg_loom_bias": "Reflex bias (loom_persist)",
        "leg_odor_l":    "Left antenna odor",
        "leg_odor_r":    "Right antenna odor",
        "loom_signal":   "Looming signal (a.u.)",
        "reflex_bias":   "Reflex bias",
        "odor_raw":      "Odor intensity (raw)",
        "odor_field_label": "Odor intensity (path-distance, log scale)",
        "odor_asym_label":  "Odor asymmetry R-L (normalised)",
        "t13":              "Odor-movement correlation",
        "odor_total":       "Total odor intensity (L+R)",
        "odor_asym_norm":   "Odor asymmetry R-L (norm.)",
        "turn_signal":      "Turn signal (ctrl_L - ctrl_R)",
        "heading_rate":     "Heading change rate (rad/step)",
        "corr_scatter":     "Odor asymmetry vs turn signal",
        "leg_odor_total":   "Total odor (L+R)",
        "leg_dist":         "Distance to food (mm)",
        "leg_asym":         "Odor asymmetry R-L (norm.)",
        "leg_turn":         "Turn signal (ctrl_L - ctrl_R)",
        "corr_label":       "Pearson r = {r:.3f}",
        "nav_label":        "Navigation phase",
    },
    "FR": {
        "time_s":            "Temps de simulation (s)",
        "dist_food":         "dist. nourriture (mm)",
        "mean_rate_hz":      "Taux moyen (Hz)",
        "neuron_count":      "Nombre de neurones",
        "ctrl_amplitude":    "Amplitude de controle",
        "asymmetry":         "Asymetrie",
        "brain_to_body":     "[cerveau -> corps]",
        "body_to_brain":     "[corps -> cerveau]",
        "odor_norm":         "Intensite olfactive (normalisee)",
        "olf_rate":          "Taux moyen olfactif (Hz)",
        "asc_rate_label":    "Taux ascendant (Hz)\n[corps -> cerveau]",
        "asym_label":        "Asymetrie\n[cerveau -> corps]",
        "spike_count":       "Spikes / pas",
        "dn_lr_diff":        "DN diff G/D",
        "mean_rate_label":   "Taux moyen (Hz)",
        "dn_left_leg":       "DN gauche (inverse)",

        # circuit labels
        "c_asc":   "Ascendants / proprio",
        "c_olf":   "Olfactifs (echantillon)",
        "c_sez":   "SEZ / alimentation",
        "c_dnl":   "DN gauche",
        "c_dnr":   "DN droit",
        "c_olf_s": "Olfactifs\n(echantillon)",
        "c_dn":    "DN (G+D)",

        # legend entries
        "leg_dn_left":   "DN gauche",
        "leg_dn_right":  "DN droit (inverse)",
        "leg_ctrl_l":    "ctrl gauche",
        "leg_ctrl_r":    "ctrl droit",
        "leg_dn_brain":  "DN diff G/D (cerveau)",
        "leg_ctrl_body": "ctrl_g - ctrl_d (corps)",
        "leg_odor":      "Intensite olfactive (norm.)",
        "leg_olf":       "Taux moyen olfactif (Hz)",

        # titles
        "t01": "Chronologie d'activation des circuits",
        "t02": "Raster de spikes (circuits cles)",
        "t03": "Activite des neurones descendants vs sortie motrice",
        "t04": "Couplage cerveau-corps en boucle fermee",
        "t05": "Carte de chaleur du taux de decharge (bins 100ms)",
        "t06": "Distribution du taux de decharge moyen par circuit",
        "t07": "Gradient olfactif vs activite du circuit olfactif",
        "t08": "Luminance oculaire vs taux de decharge des neurones laminaires LA>ME",
        "t_cbar": "Taux moyen (Hz)",

        "leg_vis_left":  "Oeil gauche -> lamina gauche (Hz)",
        "leg_vis_right": "Oeil droit -> lamina droit (Hz)",
        "vis_rate_hz":   "Taux laminaire (Hz)",
        "luminance":     "Luminance (norm.)",

        "t09": "Trajectoire de la mouche avec disposition des murs",
        "t10": "Reflexe de fuite : signal par oeil et biais de direction",
        "t11": "Odeur antenne gauche vs droite et direction",
        "t12": "Trajectoire superposee au gradient olfactif canalise",
        "x_mm": "Position X (mm)", "y_mm": "Position Y (mm)",
        "leg_path":      "Trajectoire",
        "leg_landmark":  "Repere visuel",
        "leg_food":      "Nourriture",
        "leg_spawn":     "Depart",
        "leg_wall":      "Mur",
        "leg_tunnel":    "Couloir",
        "leg_loom_l":    "Signal loom oeil gauche",
        "leg_loom_r":    "Signal loom oeil droit",
        "leg_loom_bias": "Biais reflexe (loom_persist)",
        "leg_odor_l":    "Odeur antenne gauche",
        "leg_odor_r":    "Odeur antenne droite",
        "loom_signal":   "Signal loom (u.a.)",
        "reflex_bias":   "Biais reflexe",
        "odor_raw":      "Intensite olfactive (brute)",
        "odor_field_label": "Intensite olfactive (distance de chemin, echelle log)",
        "odor_asym_label":  "Asymetrie olfactive D-G (normalisee)",
        "t13":              "Correlation odeur-mouvement",
        "odor_total":       "Intensite olfactive totale (G+D)",
        "odor_asym_norm":   "Asymetrie olfactive D-G (norm.)",
        "turn_signal":      "Signal de virage (ctrl_G - ctrl_D)",
        "heading_rate":     "Taux de changement de cap (rad/pas)",
        "corr_scatter":     "Asymetrie olfactive vs signal de virage",
        "leg_odor_total":   "Odeur totale (G+D)",
        "leg_dist":         "Distance nourriture (mm)",
        "leg_asym":         "Asymetrie olfactive D-G (norm.)",
        "leg_turn":         "Signal de virage (ctrl_G - ctrl_D)",
        "corr_label":       "Pearson r = {r:.3f}",
        "nav_label":        "Phase de navigation",
    },
}

# =============================================================================
#  RESOLVE INPUT FILE
# =============================================================================
sim_dir = Path(__file__).parent / "simulations"

if len(sys.argv) > 1:
    data_path = Path(sys.argv[1])
else:
    candidates = sorted(sim_dir.glob("v*_data.h5"))
    if not candidates:
        print("No vN_data.h5 found in simulations/. Run the simulation first.")
        sys.exit(1)
    data_path = candidates[-1]

print(f"Loading {data_path.name} ...")

# =============================================================================
#  LOAD DATA
# =============================================================================
with h5py.File(data_path, "r") as f:
    n_steps  = int(f["meta"].attrs["n_steps"])
    dt       = float(f["meta"].attrs["decision_interval"])
    dur      = float(f["meta"].attrs["brain_duration_s"])
    version  = int(f["meta"].attrs["version"])

    asc_rate   = f["behavior/asc_rate"][:]
    lr_diff    = f["behavior/lr_diff"][:]
    dist       = f["behavior/dist_to_food"][:]
    dn_l       = f["behavior/dn_left_count"][:]
    dn_r       = f["behavior/dn_right_count"][:]
    ctrl_left  = f["behavior/ctrl_left"][:]
    ctrl_right = f["behavior/ctrl_right"][:]
    odor_norm  = f["behavior/odor_norm"][:]
    is_feeding = f["behavior/is_feeding"][:]

    # Visual lamina rates (present only in simulations with enable_vision=True)
    _has_vision = "behavior/vis_rate_left" in f
    if _has_vision:
        vis_rate_left  = f["behavior/vis_rate_left"][:]
        vis_rate_right = f["behavior/vis_rate_right"][:]
    else:
        vis_rate_left = vis_rate_right = None

    # Looming reflex + trajectory (present in simulations with landmarks)
    _has_loom = "behavior/loom_bias" in f
    if _has_loom:
        loom_sig_l  = f["behavior/loom_signal_l"][:]
        loom_sig_r  = f["behavior/loom_signal_r"][:]
        loom_bias   = f["behavior/loom_bias"][:]
        fly_x       = f["behavior/fly_x"][:]
        fly_y       = f["behavior/fly_y"][:]
        fly_heading = f["behavior/fly_heading"][:]
        odor_left   = f["behavior/odor_left"][:]
        odor_right  = f["behavior/odor_right"][:]
        odor_asym   = f["behavior/odor_asym"][:] if "behavior/odor_asym" in f else None
    else:
        loom_sig_l = loom_sig_r = loom_bias = None
        fly_x = fly_y = fly_heading = None
        odor_left = odor_right = odor_asym = None

    # Channeled odor field (Dijkstra grid) — present in zigzag layout runs
    _has_odor_field = "odor_field" in f
    if _has_odor_field:
        _of        = f["odor_field"]
        odor_field = _of["field"][:]    # (NX, NY) float32
        odor_xs    = _of["xs"][:]       # world x coords
        odor_ys    = _of["ys"][:]       # world y coords
        odor_blk   = _of["blocked"][:].astype(bool)  # (NX, NY) wall mask
        food_xy    = (float(_of.attrs["food_x"]), float(_of.attrs["food_y"]))
    else:
        odor_field = odor_xs = odor_ys = odor_blk = None
        food_xy    = (18.0, 12.0)   # legacy default

    def load_spikes(grp):
        return (grp["times"][:], grp["idx"][:], int(grp.attrs["n_neurons"]))

    sp_dnl = load_spikes(f["spikes/dn_left"])
    sp_dnr = load_spikes(f["spikes/dn_right"])
    sp_sez = load_spikes(f["spikes/sez"])
    sp_asc = load_spikes(f["spikes/ascending"])
    sp_olf = load_spikes(f["spikes/olfactory"])

t_axis = np.arange(n_steps) * dt

# =============================================================================
#  SHARED HELPERS
# =============================================================================
FEED_COLOR = "#ff9944"
DARK_BG    = "#0a0e14"

def _feeding_spans(is_feeding, t_axis):
    spans, in_feed, t0 = [], False, 0.0
    for i, v in enumerate(is_feeding):
        if v > 0.5 and not in_feed:
            in_feed, t0 = True, t_axis[i]
        elif v < 0.5 and in_feed:
            spans.append((t0, t_axis[i]))
            in_feed = False
    if in_feed:
        spans.append((t0, t_axis[-1]))
    return spans

def _mean_rate_over_time(times, n_neurons, t_axis, dt):
    bins = np.append(t_axis, t_axis[-1] + dt)
    counts, _ = np.histogram(times, bins=bins)
    return counts / (n_neurons * dt)

def _binned_rate(times, n_neurons, bins):
    counts, _ = np.histogram(times, bins=bins)
    return counts / (n_neurons * (bins[1] - bins[0]))

def _mean_rates(times, idxs, n_neurons, dur):
    rates = np.zeros(n_neurons)
    for i in range(n_neurons):
        rates[i] = (idxs == i).sum() / dur
    return rates

def _raster_sample(times, idxs, n_neurons, max_n):
    if n_neurons <= max_n:
        return times, idxs
    keep = np.where(idxs < max_n)[0]
    return times[keep], idxs[keep]

def _style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="grey", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

feed_spans = _feeding_spans(is_feeding, t_axis)

MAX_RASTER_NEURONS = 150

# =============================================================================
#  GENERATE PLOTS PER LANGUAGE
# =============================================================================
base_dir = Path(__file__).parent / "plots" / f"v{version}"

for lang, T in LANGS.items():
    out_dir = base_dir / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{lang}] -> {out_dir}/")

    def _save(fig, name):
        path = out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  {name}")

    # -------------------------------------------------------------------------
    #  01  Circuit activation timeline
    # -------------------------------------------------------------------------
    circuit_data = [
        (T["c_asc"], sp_asc, "#4ab8cc"),
        (T["c_olf"], sp_olf, "#ff55bb"),
        (T["c_sez"], sp_sez, "#ff9944"),
        (T["c_dnl"], sp_dnl, "#44cc66"),
        (T["c_dnr"], sp_dnr, "#88ff44"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True, facecolor=DARK_BG)
    fig.suptitle(f"v{version} -- {T['t01']}", color="white", fontsize=12, y=0.99)
    for ax, (label, (times, _, n_neu), color) in zip(axes, circuit_data):
        rate = _mean_rate_over_time(times, n_neu, t_axis, dt)
        ax.fill_between(t_axis, rate, alpha=0.6, color=color)
        ax.plot(t_axis, rate, color=color, linewidth=0.8)
        for t0, t1 in feed_spans:
            ax.axvspan(t0, t1, color=FEED_COLOR, alpha=0.2)
        ax.set_ylabel(label, color=color, fontsize=8)
        _style_ax(ax)
    axes[-1].set_xlabel(T["time_s"], color="grey", fontsize=9)
    ax2 = axes[0].twinx()
    ax2.plot(t_axis, dist, color="white", linewidth=0.7, alpha=0.5, linestyle="--")
    ax2.set_ylabel(T["dist_food"], color="grey", fontsize=7)
    ax2.tick_params(colors="grey", labelsize=7)
    fig.tight_layout()
    _save(fig, "01_circuit_timeline.png")

    # -------------------------------------------------------------------------
    #  02  Raster plot
    # -------------------------------------------------------------------------
    circuits_raster = [
        (T["c_dnl"], sp_dnl, "#44cc66"),
        (T["c_dnr"], sp_dnr, "#88ff44"),
        (T["c_sez"], sp_sez, "#ff9944"),
        (T["c_asc"], sp_asc, "#4ab8cc"),
        (T["c_olf"], sp_olf, "#ff55bb"),
    ]
    fig, axes = plt.subplots(len(circuits_raster), 1, figsize=(14, 9),
                             sharex=True, facecolor=DARK_BG)
    fig.suptitle(f"v{version} -- {T['t02']}", color="white", fontsize=12, y=0.99)
    for ax, (label, (times, idxs, n_neu), color) in zip(axes, circuits_raster):
        t_s, i_s = _raster_sample(times, idxs, n_neu, MAX_RASTER_NEURONS)
        ax.scatter(t_s, i_s, s=0.4, c=color, alpha=0.6, linewidths=0)
        for t0, t1 in feed_spans:
            ax.axvspan(t0, t1, color=FEED_COLOR, alpha=0.15)
        ax.set_ylabel(label, color=color, fontsize=8)
        ax.set_ylim(-1, min(n_neu, MAX_RASTER_NEURONS))
        _style_ax(ax)
    axes[-1].set_xlabel(T["time_s"], color="grey", fontsize=9)
    fig.tight_layout()
    _save(fig, "02_raster_circuits.png")

    # -------------------------------------------------------------------------
    #  03  DN turning coupling
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True, facecolor=DARK_BG)
    fig.suptitle(f"v{version} -- {T['t03']}", color="white", fontsize=12, y=0.99)

    axes[0].fill_between(t_axis, dn_l,  alpha=0.5, color="#44cc66", label=T["leg_dn_left"])
    axes[0].fill_between(t_axis, -dn_r, alpha=0.5, color="#88ff44", label=T["leg_dn_right"])
    axes[0].axhline(0, color="#333344", linewidth=0.5)
    axes[0].set_ylabel(T["spike_count"], color="grey", fontsize=8)
    axes[0].legend(fontsize=7, facecolor="#111122", labelcolor="white", loc="upper right")

    axes[1].plot(t_axis, lr_diff, color="#ff88cc", linewidth=1.0)
    axes[1].axhline(0, color="#333344", linewidth=0.5)
    axes[1].set_ylabel(T["dn_lr_diff"], color="grey", fontsize=8)

    axes[2].plot(t_axis, ctrl_left,  color="#44cc66", linewidth=1.0, label=T["leg_ctrl_l"])
    axes[2].plot(t_axis, ctrl_right, color="#88ff44", linewidth=1.0, label=T["leg_ctrl_r"], alpha=0.8)
    axes[2].set_ylabel(T["ctrl_amplitude"], color="grey", fontsize=8)
    axes[2].legend(fontsize=7, facecolor="#111122", labelcolor="white", loc="upper right")

    for ax in axes:
        for t0, t1 in feed_spans:
            ax.axvspan(t0, t1, color=FEED_COLOR, alpha=0.15)
        _style_ax(ax)
    axes[-1].set_xlabel(T["time_s"], color="grey", fontsize=9)
    fig.tight_layout()
    _save(fig, "03_dn_turning_coupling.png")

    # -------------------------------------------------------------------------
    #  04  Brain-body coupling
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, facecolor=DARK_BG)
    fig.suptitle(f"v{version} -- {T['t04']}", color="white", fontsize=12, y=0.99)

    axes[0].plot(t_axis, asc_rate, color="#4ab8cc", linewidth=1.0)
    axes[0].set_ylabel(T["asc_rate_label"], color="#4ab8cc", fontsize=8)
    axes[0].set_ylim(0, None)

    net_turn = ctrl_left - ctrl_right
    axes[1].plot(t_axis, lr_diff,  color="#ff88cc", linewidth=1.0, label=T["leg_dn_brain"])
    axes[1].plot(t_axis, net_turn, color="#44cc66", linewidth=1.0, alpha=0.8,
                 label=T["leg_ctrl_body"])
    axes[1].axhline(0, color="#333344", linewidth=0.5)
    axes[1].set_ylabel(T["asym_label"], color="grey", fontsize=8)
    axes[1].legend(fontsize=7, facecolor="#111122", labelcolor="white")

    for ax in axes:
        for t0, t1 in feed_spans:
            ax.axvspan(t0, t1, color=FEED_COLOR, alpha=0.15)
        _style_ax(ax)
    axes[-1].set_xlabel(T["time_s"], color="grey", fontsize=9)
    fig.tight_layout()
    _save(fig, "04_brain_body_coupling.png")

    # -------------------------------------------------------------------------
    #  05  Population heatmap
    # -------------------------------------------------------------------------
    BIN_S = 0.1
    bins  = np.arange(0, dur + BIN_S, BIN_S)
    heatmap_circuits = [
        (T["c_asc"], sp_asc),
        (T["c_olf"], sp_olf),
        (T["c_sez"], sp_sez),
        (T["c_dnl"], sp_dnl),
        (T["c_dnr"], sp_dnr),
    ]
    matrix = np.array([_binned_rate(sp[0], sp[2], bins) for _, sp in heatmap_circuits])

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
    fig.suptitle(f"v{version} -- {T['t05']}", color="white", fontsize=12)
    im = ax.imshow(matrix, aspect="auto", origin="lower",
                   extent=[0, dur, -0.5, len(heatmap_circuits) - 0.5],
                   cmap="inferno", interpolation="nearest")
    ax.set_yticks(range(len(heatmap_circuits)))
    ax.set_yticklabels([c[0] for c in heatmap_circuits], color="white", fontsize=9)
    ax.set_xlabel(T["time_s"], color="grey", fontsize=9)
    ax.tick_params(colors="grey")
    ax.set_facecolor(DARK_BG)
    for t0, t1 in feed_spans:
        ax.axvspan(t0, t1, color=FEED_COLOR, alpha=0.3)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(T["t_cbar"], color="grey", fontsize=8)
    cbar.ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    fig.tight_layout()
    _save(fig, "05_population_heatmap.png")

    # -------------------------------------------------------------------------
    #  06  Firing rate distribution
    # -------------------------------------------------------------------------
    dist_circuits = [
        (T["c_dn"],    np.concatenate([sp_dnl[0], sp_dnr[0]]),
                       np.concatenate([sp_dnl[1], sp_dnr[1]]),
                       sp_dnl[2] + sp_dnr[2], "#44cc66"),
        (T["c_sez"],   sp_sez[0], sp_sez[1], sp_sez[2],  "#ff9944"),
        (T["c_olf_s"], sp_olf[0], sp_olf[1], sp_olf[2],  "#ff55bb"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=DARK_BG)
    fig.suptitle(f"v{version} -- {T['t06']}", color="white", fontsize=12)
    for ax, (label, times, idxs, n_neu, color) in zip(axes, dist_circuits):
        rates = _mean_rates(times, idxs, n_neu, dur)
        ax.hist(rates, bins=40, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(rates.mean(), color="white", linewidth=1.0, linestyle="--",
                   label=f"mean={rates.mean():.1f}Hz")
        ax.set_title(label, color=color, fontsize=9)
        ax.set_xlabel(T["mean_rate_label"], color="grey", fontsize=8)
        ax.set_ylabel(T["neuron_count"], color="grey", fontsize=8)
        ax.legend(fontsize=7, facecolor="#111122", labelcolor="white")
        _style_ax(ax)
    fig.tight_layout()
    _save(fig, "06_firing_rate_distribution.png")

    # -------------------------------------------------------------------------
    #  07  Odor vs olfactory response
    # -------------------------------------------------------------------------
    olf_rate = _mean_rate_over_time(sp_olf[0], sp_olf[2], t_axis, dt)

    fig, ax1 = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
    fig.suptitle(f"v{version} -- {T['t07']}", color="white", fontsize=12)
    ax1.set_facecolor(DARK_BG)
    ax1.fill_between(t_axis, odor_norm, alpha=0.4, color="#88ccff")
    ax1.plot(t_axis, odor_norm, color="#88ccff", linewidth=0.8, label=T["leg_odor"])
    ax1.set_ylabel(T["odor_norm"], color="#88ccff", fontsize=9)
    ax1.tick_params(colors="grey", labelsize=7)

    ax2 = ax1.twinx()
    ax2.plot(t_axis, olf_rate, color="#ff55bb", linewidth=1.2, label=T["leg_olf"])
    ax2.set_ylabel(T["olf_rate"], color="#ff55bb", fontsize=9)
    ax2.tick_params(colors="grey", labelsize=7)

    for t0, t1 in feed_spans:
        ax1.axvspan(t0, t1, color=FEED_COLOR, alpha=0.2)
    ax1.set_xlabel(T["time_s"], color="grey", fontsize=9)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               facecolor="#111122", labelcolor="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333344")
    fig.tight_layout()
    _save(fig, "07_odor_olfactory_response.png")

    # -------------------------------------------------------------------------
    #  08  Compound-eye luminance vs LA>ME lamina firing rate
    # -------------------------------------------------------------------------
    if vis_rate_left is not None:
        VIS_MIN, VIS_MAX = 20.0, 150.0
        lum_left  = (vis_rate_left  - VIS_MIN) / (VIS_MAX - VIS_MIN)
        lum_right = (vis_rate_right - VIS_MIN) / (VIS_MAX - VIS_MIN)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                        sharex=True, facecolor=DARK_BG)
        fig.suptitle(f"v{version} -- {T['t08']}", color="white", fontsize=12)

        # top panel: firing rates
        ax1.plot(t_axis, vis_rate_left,  color="#ffee00", linewidth=1.2,
                 label=T["leg_vis_left"])
        ax1.plot(t_axis, vis_rate_right, color="#ffaa00", linewidth=1.2,
                 linestyle="--", label=T["leg_vis_right"])
        ax1.set_ylabel(T["vis_rate_hz"], color="grey", fontsize=9)
        ax1.legend(fontsize=8, facecolor="#111122", labelcolor="white")
        _style_ax(ax1)

        # bottom panel: normalised luminance
        ax2.fill_between(t_axis, lum_left,  alpha=0.35, color="#ffee00")
        ax2.fill_between(t_axis, lum_right, alpha=0.35, color="#ffaa00")
        ax2.plot(t_axis, lum_left,  color="#ffee00", linewidth=0.9)
        ax2.plot(t_axis, lum_right, color="#ffaa00", linewidth=0.9, linestyle="--")
        ax2.set_ylabel(T["luminance"], color="grey", fontsize=9)
        ax2.set_xlabel(T["time_s"],    color="grey", fontsize=9)
        _style_ax(ax2)

        for ax in (ax1, ax2):
            for t0, t1 in feed_spans:
                ax.axvspan(t0, t1, color=FEED_COLOR, alpha=0.2)

        fig.tight_layout()
        _save(fig, "08_visual_lamina_response.png")
    else:
        print(f"  08_visual_lamina_response.png -- SKIPPED (no vision data in this HDF5)")

    # -------------------------------------------------------------------------
    #  09  Fly trajectory with wall layout
    # -------------------------------------------------------------------------
    if fly_x is not None and fly_x.any():
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK_BG)
        fig.suptitle(f"v{version} -- {T['t09']}", color="white", fontsize=12)
        ax.set_facecolor(DARK_BG)

        # Draw walls from blocked mask (if available) or skip
        if odor_blk is not None:
            # Convert boolean grid to wall rectangles for efficiency
            wall_color = "#5c3010"
            # sample blocked mask at 1mm resolution for rendering
            step_x = max(1, int(1.0 / (odor_xs[1] - odor_xs[0])))
            step_y = max(1, int(1.0 / (odor_ys[1] - odor_ys[0])))
            for ix in range(0, len(odor_xs), step_x):
                for iy in range(0, len(odor_ys), step_y):
                    if odor_blk[ix, iy]:
                        rx = odor_xs[ix] - (odor_xs[1]-odor_xs[0])*0.5
                        ry = odor_ys[iy] - (odor_ys[1]-odor_ys[0])*0.5
                        rw = (odor_xs[1]-odor_xs[0]) * step_x
                        rh = (odor_ys[1]-odor_ys[0]) * step_y
                        ax.add_patch(mpatches.Rectangle(
                            (rx, ry), rw, rh,
                            color=wall_color, zorder=2))
            wall_patch = mpatches.Patch(color=wall_color, label=T["leg_wall"])

        # Tunnel corridor annotation (zigzag layout)
        tunnel = mpatches.Rectangle(
            (8.0, 6.0), 6.0, 4.0,
            linewidth=1, edgecolor="#44aaff", facecolor="none",
            linestyle="--", zorder=3, label=T["leg_tunnel"])
        ax.add_patch(tunnel)

        # Trajectory colored by time
        points   = np.array([fly_x, fly_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(vmin=0, vmax=len(fly_x))
        lc   = LineCollection(segments, cmap="plasma", norm=norm,
                              linewidth=2.0, alpha=0.9, zorder=4)
        lc.set_array(np.arange(len(fly_x)))
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, label=T["time_s"], fraction=0.03, pad=0.02)
        cb.ax.yaxis.set_tick_params(color="grey")
        cb.ax.tick_params(colors="grey", labelsize=7)

        # Heading arrows every 20 steps
        for i in range(0, len(fly_x), 20):
            ax.annotate("", xy=(fly_x[i] + 0.7*np.cos(fly_heading[i]),
                                fly_y[i] + 0.7*np.sin(fly_heading[i])),
                        xytext=(fly_x[i], fly_y[i]),
                        arrowprops=dict(arrowstyle="->", color="white", lw=0.7),
                        zorder=5)

        # Food + spawn markers
        ax.plot(*food_xy, "*", color=FEED_COLOR, markersize=16,
                label=T["leg_food"], zorder=7)
        ax.plot(fly_x[0], fly_y[0], "^", color="cyan", markersize=9,
                label=T["leg_spawn"], zorder=7)

        # Feeding positions
        feed_steps = np.where(is_feeding > 0.5)[0]
        if len(feed_steps):
            ax.scatter(fly_x[feed_steps], fly_y[feed_steps],
                       c=FEED_COLOR, s=14, zorder=8, alpha=0.8)

        # Legend
        handles = [tunnel]
        if odor_blk is not None:
            handles.append(wall_patch)
        extra_h, extra_l = ax.get_legend_handles_labels()
        ax.legend(handles + extra_h, [t.get_label() for t in handles] + extra_l,
                  fontsize=8, facecolor="#111122", labelcolor="white")

        ax.set_xlabel(T["x_mm"], color="grey", fontsize=9)
        ax.set_ylabel(T["y_mm"], color="grey", fontsize=9)
        x_lo = min(-1, float(fly_x.min()) - 1)
        x_hi = max(food_xy[0] + 2, float(fly_x.max()) + 1)
        y_lo = min(-16, float(fly_y.min()) - 1)
        y_hi = max(food_xy[1] + 2, float(fly_y.max()) + 1)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.tick_params(colors="grey", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        fig.tight_layout()
        _save(fig, "09_trajectory.png")
    else:
        print(f"  09_trajectory.png -- SKIPPED (no position data in this HDF5)")

    # -------------------------------------------------------------------------
    #  10  Looming reflex: per-eye signal and steering bias
    # -------------------------------------------------------------------------
    if loom_bias is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                        sharex=True, facecolor=DARK_BG)
        fig.suptitle(f"v{version} -- {T['t10']}", color="white", fontsize=12)

        # top: per-eye looming signal
        ax1.plot(t_axis, loom_sig_l, color="#ffee00", linewidth=1.2,
                 label=T["leg_loom_l"])
        ax1.plot(t_axis, loom_sig_r, color="#ff6600", linewidth=1.2,
                 linestyle="--", label=T["leg_loom_r"])
        ax1.set_ylabel(T["loom_signal"], color="grey", fontsize=9)
        ax1.legend(fontsize=8, facecolor="#111122", labelcolor="white")
        _style_ax(ax1)

        # bottom: persisted reflex bias
        ax2.fill_between(t_axis, loom_bias, 0, where=loom_bias < 0,
                         color="#ff4444", alpha=0.4, label="turn right")
        ax2.fill_between(t_axis, loom_bias, 0, where=loom_bias > 0,
                         color="#44aaff", alpha=0.4, label="turn left")
        ax2.plot(t_axis, loom_bias, color="white", linewidth=0.8,
                 label=T["leg_loom_bias"])
        ax2.axhline(0, color="#555566", linewidth=0.6)
        ax2.set_ylabel(T["reflex_bias"], color="grey", fontsize=9)
        ax2.set_xlabel(T["time_s"], color="grey", fontsize=9)
        ax2.legend(fontsize=8, facecolor="#111122", labelcolor="white")
        _style_ax(ax2)

        for ax in (ax1, ax2):
            for t0, t1 in feed_spans:
                ax.axvspan(t0, t1, color=FEED_COLOR, alpha=0.2)

        fig.tight_layout()
        _save(fig, "10_looming_reflex.png")
    else:
        print(f"  10_looming_reflex.png -- SKIPPED (no loom data in this HDF5)")

    # -------------------------------------------------------------------------
    #  11  Left vs right antenna odor + steering asymmetry
    # -------------------------------------------------------------------------
    if odor_left is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                        sharex=True, facecolor=DARK_BG)
        fig.suptitle(f"v{version} -- {T['t11']}", color="white", fontsize=12)

        # top: raw antenna odor L/R
        ax1.fill_between(t_axis, odor_left,  alpha=0.35, color="#44aaff")
        ax1.fill_between(t_axis, odor_right, alpha=0.35, color="#ff6688")
        ax1.plot(t_axis, odor_left,  color="#44aaff", linewidth=1.0,
                 label=T["leg_odor_l"])
        ax1.plot(t_axis, odor_right, color="#ff6688", linewidth=1.0,
                 linestyle="--", label=T["leg_odor_r"])
        ax1.set_ylabel(T["odor_raw"], color="grey", fontsize=9)
        ax1.legend(fontsize=8, facecolor="#111122", labelcolor="white")
        _style_ax(ax1)

        # bottom: L-R asymmetry driving turn + ctrl diff
        # use stored odor_asym if available (properly normalised at steering time)
        _asym_plot = odor_asym if odor_asym is not None else (odor_right - odor_left)
        ctrl_diff  = ctrl_left - ctrl_right   # positive = turning right
        ax2.plot(t_axis, _asym_plot / (np.abs(_asym_plot).max() + 1e-9),
                 color="#aaffaa", linewidth=1.0, label="odor R-L (norm.)")
        ax2.plot(t_axis, ctrl_diff, color="#ff9944", linewidth=1.2,
                 label="ctrl L-R")
        ax2.axhline(0, color="#555566", linewidth=0.6)
        ax2.set_ylabel(T["asymmetry"], color="grey", fontsize=9)
        ax2.set_xlabel(T["time_s"], color="grey", fontsize=9)
        ax2.legend(fontsize=8, facecolor="#111122", labelcolor="white")
        _style_ax(ax2)

        for ax in (ax1, ax2):
            for t0, t1 in feed_spans:
                ax.axvspan(t0, t1, color=FEED_COLOR, alpha=0.2)

        fig.tight_layout()
        _save(fig, "11_odor_asymmetry.png")
    else:
        print(f"  11_odor_asymmetry.png -- SKIPPED (no odor L/R data in this HDF5)")

    # -------------------------------------------------------------------------
    #  12  Trajectory overlaid on channeled odor gradient heatmap
    # -------------------------------------------------------------------------
    if odor_field is not None and fly_x is not None and fly_x.any():
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize, LogNorm
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(11, 9), facecolor=DARK_BG)
        fig.suptitle(f"v{version} -- {T['t12']}", color="white", fontsize=12)
        ax.set_facecolor(DARK_BG)

        # Odor field as heatmap — transpose so axes are (x, y) -> (col, row)
        # pcolormesh expects (NY, NX) with x=xs, y=ys
        field_T = odor_field.T   # (NY, NX)
        vmin_nz = field_T[field_T > 0].min() if (field_T > 0).any() else 1e-3
        pcm = ax.pcolormesh(
            odor_xs, odor_ys, field_T,
            norm=LogNorm(vmin=max(vmin_nz, 0.01), vmax=field_T.max()),
            cmap="inferno", shading="auto", zorder=1, alpha=0.85,
        )
        cb = fig.colorbar(pcm, ax=ax, label=T["odor_field_label"],
                          fraction=0.03, pad=0.02)
        cb.ax.tick_params(colors="grey", labelsize=7)
        cb.ax.yaxis.label.set_color("grey")

        # Walls: draw blocked cells as dark overlay
        wall_img = np.zeros((*field_T.shape, 4))  # RGBA
        wall_img[odor_blk.T] = [0.12, 0.06, 0.02, 0.95]
        ax.imshow(
            wall_img,
            extent=[odor_xs[0], odor_xs[-1], odor_ys[0], odor_ys[-1]],
            origin="lower", aspect="auto", zorder=2,
        )

        # Tunnel corridor outline
        tunnel = mpatches.Rectangle(
            (8.0, 6.0), 6.0, 4.0,
            linewidth=1.2, edgecolor="#44aaff", facecolor="none",
            linestyle="--", zorder=3, label=T["leg_tunnel"])
        ax.add_patch(tunnel)

        # Trajectory colored by time
        points   = np.array([fly_x, fly_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm_t   = Normalize(vmin=0, vmax=len(fly_x))
        lc = LineCollection(segments, cmap="cool", norm=norm_t,
                            linewidth=2.5, alpha=0.95, zorder=4)
        lc.set_array(np.arange(len(fly_x)))
        ax.add_collection(lc)
        cb2 = fig.colorbar(lc, ax=ax, label=T["time_s"],
                           fraction=0.025, pad=0.07)
        cb2.ax.tick_params(colors="grey", labelsize=7)
        cb2.ax.yaxis.label.set_color("grey")

        # Heading arrows every 20 steps
        for i in range(0, len(fly_x), 20):
            ax.annotate("", xy=(fly_x[i] + 0.7*np.cos(fly_heading[i]),
                                fly_y[i] + 0.7*np.sin(fly_heading[i])),
                        xytext=(fly_x[i], fly_y[i]),
                        arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
                        zorder=5)

        # Food + spawn
        ax.plot(*food_xy, "*", color=FEED_COLOR, markersize=18,
                label=T["leg_food"], zorder=6)
        ax.plot(fly_x[0], fly_y[0], "^", color="cyan", markersize=10,
                label=T["leg_spawn"], zorder=6)

        # Feeding positions
        feed_steps = np.where(is_feeding > 0.5)[0]
        if len(feed_steps):
            ax.scatter(fly_x[feed_steps], fly_y[feed_steps],
                       c=FEED_COLOR, s=18, zorder=7, alpha=0.9)

        handles, labels = ax.get_legend_handles_labels()
        handles.append(tunnel)
        labels.append(T["leg_tunnel"])
        ax.legend(handles, labels, fontsize=8,
                  facecolor="#111122", labelcolor="white")

        ax.set_xlabel(T["x_mm"], color="grey", fontsize=9)
        ax.set_ylabel(T["y_mm"], color="grey", fontsize=9)
        ax.set_xlim(odor_xs[0], odor_xs[-1])
        ax.set_ylim(odor_ys[0], odor_ys[-1])
        ax.set_aspect("equal")
        ax.tick_params(colors="grey", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        fig.tight_layout()
        _save(fig, "12_odor_field_trajectory.png")
    else:
        print(f"  12_odor_field_trajectory.png -- SKIPPED (no odor field or position data)")

    # -------------------------------------------------------------------------
    #  13  Odor-movement correlation
    #      Panel 1: total odor + distance to food over time (what the fly sensed)
    #      Panel 2: odor asymmetry vs turn signal over time (how it steered)
    #      Panel 3: scatter odor_asym vs turn_signal colored by time (correlation)
    # -------------------------------------------------------------------------
    if odor_left is not None and fly_x is not None and fly_x.any():
        from matplotlib.colors import Normalize
        import matplotlib.gridspec as gridspec

        _asym  = odor_asym if odor_asym is not None else (
            (odor_right - odor_left) / (odor_right + odor_left + 1e-9))
        _total = odor_left + odor_right
        _turn  = ctrl_left - ctrl_right   # positive = turning left
        _asym_n = _asym / (np.abs(_asym).max() + 1e-9)
        _turn_n = _turn / (np.abs(_turn).max() + 1e-9)

        # heading change rate (finite diff, wrap-safe)
        _hdiff = np.diff(fly_heading, prepend=fly_heading[0])
        _hdiff = ((_hdiff + np.pi) % (2*np.pi)) - np.pi   # wrap to [-pi, pi]

        # Pearson correlation (odor asym vs turn signal, walking steps only)
        _walk_mask = is_feeding < 0.5
        _r_asym_turn = float(np.corrcoef(
            _asym_n[_walk_mask], _turn_n[_walk_mask])[0, 1])
        _r_asym_head = float(np.corrcoef(
            _asym_n[_walk_mask], _hdiff[_walk_mask])[0, 1])

        fig = plt.figure(figsize=(14, 12), facecolor=DARK_BG)
        fig.suptitle(f"v{version} -- {T['t13']}", color="white", fontsize=13, y=0.99)
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ── Panel 1 (top, full width): total odor + distance over time ────────
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(DARK_BG)
        ax1.fill_between(t_axis, _total, alpha=0.3, color="#ff88cc")
        ax1.plot(t_axis, _total, color="#ff88cc", linewidth=1.2,
                 label=T["leg_odor_total"])
        for t0, t1 in feed_spans:
            ax1.axvspan(t0, t1, color=FEED_COLOR, alpha=0.25)
        ax1.set_ylabel(T["odor_total"], color="grey", fontsize=9)
        ax1_r = ax1.twinx()
        ax1_r.plot(t_axis, dist, color="white", linewidth=0.9,
                   linestyle="--", alpha=0.7, label=T["leg_dist"])
        ax1_r.set_ylabel(T["leg_dist"], color="grey", fontsize=8)
        ax1_r.tick_params(colors="grey", labelsize=7)
        ax1.set_xlabel(T["time_s"], color="grey", fontsize=9)
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax1_r.get_legend_handles_labels()
        ax1.legend(lines1+lines2, lab1+lab2, fontsize=8,
                   facecolor="#111122", labelcolor="white")
        _style_ax(ax1)

        # ── Panel 2 (middle, full width): asym vs turn signal over time ───────
        ax2 = fig.add_subplot(gs[1, :])
        ax2.set_facecolor(DARK_BG)
        ax2.plot(t_axis, _asym_n, color="#44aaff", linewidth=1.2,
                 label=T["leg_asym"])
        ax2.plot(t_axis, _turn_n, color="#ff9944", linewidth=1.2,
                 linestyle="--", label=T["leg_turn"])
        ax2.axhline(0, color="#555566", linewidth=0.5)
        for t0, t1 in feed_spans:
            ax2.axvspan(t0, t1, color=FEED_COLOR, alpha=0.25)
        ax2.set_ylabel(T["odor_asym_norm"], color="grey", fontsize=9)
        ax2.set_xlabel(T["time_s"], color="grey", fontsize=9)
        ax2.legend(fontsize=8, facecolor="#111122", labelcolor="white")
        ax2.set_title(T["corr_label"].format(r=_r_asym_turn),
                      color="#aaaaaa", fontsize=9)
        _style_ax(ax2)

        # ── Panel 3 (bottom-left): scatter asym vs turn colored by time ───────
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.set_facecolor(DARK_BG)
        sc = ax3.scatter(_asym_n[_walk_mask], _turn_n[_walk_mask],
                         c=t_axis[_walk_mask], cmap="plasma",
                         s=8, alpha=0.6, zorder=2)
        # regression line
        _x_fit = np.linspace(_asym_n.min(), _asym_n.max(), 100)
        _coeffs = np.polyfit(_asym_n[_walk_mask], _turn_n[_walk_mask], 1)
        ax3.plot(_x_fit, np.polyval(_coeffs, _x_fit),
                 color="white", linewidth=1.2, linestyle="--", alpha=0.7)
        ax3.axhline(0, color="#444455", linewidth=0.5)
        ax3.axvline(0, color="#444455", linewidth=0.5)
        cb3 = fig.colorbar(sc, ax=ax3, label=T["time_s"], fraction=0.05, pad=0.02)
        cb3.ax.tick_params(colors="grey", labelsize=7)
        cb3.ax.yaxis.label.set_color("grey")
        ax3.set_xlabel(T["odor_asym_norm"], color="grey", fontsize=9)
        ax3.set_ylabel(T["turn_signal"],    color="grey", fontsize=9)
        ax3.set_title(T["corr_scatter"],    color="#aaaaaa", fontsize=9)
        ax3.text(0.05, 0.92, T["corr_label"].format(r=_r_asym_turn),
                 transform=ax3.transAxes, color="white", fontsize=9)
        _style_ax(ax3)

        # ── Panel 4 (bottom-right): asym vs heading change rate ───────────────
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.set_facecolor(DARK_BG)
        sc2 = ax4.scatter(_asym_n[_walk_mask], _hdiff[_walk_mask],
                          c=t_axis[_walk_mask], cmap="plasma",
                          s=8, alpha=0.6, zorder=2)
        _coeffs2 = np.polyfit(_asym_n[_walk_mask], _hdiff[_walk_mask], 1)
        ax4.plot(_x_fit, np.polyval(_coeffs2, _x_fit),
                 color="white", linewidth=1.2, linestyle="--", alpha=0.7)
        ax4.axhline(0, color="#444455", linewidth=0.5)
        ax4.axvline(0, color="#444455", linewidth=0.5)
        cb4 = fig.colorbar(sc2, ax=ax4, label=T["time_s"], fraction=0.05, pad=0.02)
        cb4.ax.tick_params(colors="grey", labelsize=7)
        cb4.ax.yaxis.label.set_color("grey")
        ax4.set_xlabel(T["odor_asym_norm"], color="grey", fontsize=9)
        ax4.set_ylabel(T["heading_rate"],   color="grey", fontsize=9)
        ax4.set_title(T["corr_scatter"],    color="#aaaaaa", fontsize=9)
        ax4.text(0.05, 0.92, T["corr_label"].format(r=_r_asym_head),
                 transform=ax4.transAxes, color="white", fontsize=9)
        _style_ax(ax4)

        _save(fig, "13_odor_movement_correlation.png")
    else:
        print(f"  13_odor_movement_correlation.png -- SKIPPED (no odor or position data)")

# =============================================================================
print(f"\nAll plots saved to {base_dir}/EN/  and  {base_dir}/FR/")

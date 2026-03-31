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
        "t_cbar": "Mean rate (Hz)",
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
        "t_cbar": "Taux moyen (Hz)",
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

# =============================================================================
print(f"\nAll plots saved to {base_dir}/EN/  and  {base_dir}/FR/")

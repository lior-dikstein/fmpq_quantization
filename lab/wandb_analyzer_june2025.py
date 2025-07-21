import numpy as np
import pickle

import os

import wandb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

matplotlib.use("TkAgg")
plt.ion()  # turn on interactive mode

ENTITY = "sony-semi-ml"
PROJECT = "MpPerCh_July2025"
SAVE_PATH = '/Vols/vol_design/tools/swat/users/liord/temp/runs_cache.pkl'
LEGEND_REPLACER = {
    'mp-ch': 'MP-per-ch',
    'low rank': 'MLoRQ',
    'mp-ch-simd32': 'MP-per-ch SIMD 32',
    'mp-ch-simd16': 'MP-per-ch SIMD 16',
    'mp': 'layer MP',
    'wandb_notes': 'Mixed Precision Type',
    'exp_notes': 'MP Metric Type',
    'mp_per_channel_cost': 'MP Metric Type',
}
class WandBAnalyzer:
    """
    Class to analyze and plot metrics from a Weights & Biases project,
    with local caching of run data to avoid repeated API fetches.
    """

    def __init__(self, entity: str, project: str, cache_path: str = SAVE_PATH):
        """
        Initialize WandBAnalyzer.

        Args:
            entity (str): W&B entity (username or team).
            project (str): W&B project name.
            cache_path (str): Local path to cache serialized run data.
        """
        self.entity = entity
        self.project = project
        self.cache_path = cache_path
        self.api = wandb.Api()

    def fetch_runs(self, filters: dict, force_refresh: bool = False) -> list:
        """
        Fetch runs from the W&B project with optional local caching.

        Args:
            filters (dict): Filter criteria for runs, e.g.:
                {
                    "config.weight_n_bits": {"$in": [2, 4, 8]},
                    "config.disable_feature": True
                }
            force_refresh (bool): If True, ignore existing cache and re-fetch.

        Returns:
            list: List of run data dicts, each containing 'id', 'config', and 'summary'.
        """
        # If cache exists and refresh not forced, load from cache
        if os.path.exists(self.cache_path) and not force_refresh:
            with open(self.cache_path, "rb") as f:
                runs_data = pickle.load(f)
            return runs_data

        # Otherwise, fetch from W&B API
        path = f"{self.entity}/{self.project}"
        runs = self.api.runs(path, filters=filters)

        # Serialize relevant fields from each run
        runs_data = []
        for run in runs:
            runs_data.append({
                "id": run.id,
                "config": dict(run.config),
                "summary": dict(run.summary)
            })

        # Save to local cache
        with open(self.cache_path, "wb") as f:
            pickle.dump(runs_data, f)

        return runs_data

    def extract_data(self, runs_data: list, x_field: str, y_field: str, line_field: str) -> dict:
        """
        Extract (x, y) pairs for plotting, grouping by line_field value.

        Args:
            runs_data (list): List of run data dicts from fetch_runs().
            x_field (str): Field name for x-axis, prefixed "config." or "summary.".
            y_field (str): Field name for y-axis, prefixed "config." or "summary.".
            line_field (str): Field name to group lines, prefixed "config." or "summary.".

        Returns:
            dict: Mapping from each unique line_field value to a sorted list of (x, y) tuples.
        """
        data_dict = {}

        def _get_value(run_dict: dict, field: str):
            """
            Retrieve a value from run_dict based on prefix.

            Args:
                run_dict (dict): Single run data with 'config' and 'summary' keys.
                field (str): Field string prefixed with "config." or "summary.".

            Returns:
                Any: Extracted value, or None if missing.
            """
            if field.startswith("config."):
                key = field.split(".", 1)[1]
                return run_dict["config"].get(key)
            elif field.startswith("summary."):
                key = field.split(".", 1)[1]
                return run_dict["summary"].get(key)
            return None

        for run_dict in runs_data:
            x_val = _get_value(run_dict, x_field)
            y_val = _get_value(run_dict, y_field)
            line_val = _get_value(run_dict, line_field)
            if x_val is None or y_val is None or line_val is None:
                continue
            data_dict.setdefault(line_val, []).append((x_val, y_val))

        # Sort each list of (x, y) by x value
        for key in data_dict:
            data_dict[key] = sorted(data_dict[key], key=lambda pair: pair[0])

        return data_dict

    def plot(self, data_dict: dict, x_label: str, y_label: str, title: str = None) -> None:
        """
        Plot extracted data as lines grouped by line_field values.

        Args:
            data_dict (dict): Mapping from line_field value to (x, y) lists.
            x_label (str): Label for x-axis.
            y_label (str): Label for y-axis.
            title (str, optional): Plot title.
        """
        for line_val, points in data_dict.items():
            xs, ys = zip(*points)
            plt.plot(xs, ys, marker='o', label=str(line_val))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=True)


def shade_color(color, fraction):
    """
    Lighten a color by fraction (0→1 toward white).
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * fraction)


def plot_accuracy_bar_subplots(
        runs_data: list,
        weight_field: str,
        model_field: str,
        notes_field: str,
        exps_field: str,
        accuracy_field: str,
        legend_replacer: dict = LEGEND_REPLACER,
) -> None:
    """
    Per-weight, gap-free bar plots.

    ─ skips variants with only NaNs
    ─ bars inside each model column are packed contiguously (no blank slots)
    ─ every bar has the same width, determined by the *max* variants per model
    ─ legends show only the variants that appear in the figure
    """

    # ---------------- helpers ----------------
    def _get(run, fld):
        scope, key = fld.split('.', 1)
        return run[scope].get(key)

    # ---------------- global sets ----------------
    weights = sorted({_get(r, weight_field) for r in runs_data})
    models  = sorted({_get(r, model_field)  for r in runs_data})
    notes   = sorted({_get(r, notes_field)  for r in runs_data})

    # ---------------- global y-range ----------------
    acc_vals = [_get(r, accuracy_field)
                for r in runs_data if _get(r, accuracy_field) is not None]
    gmin, gmax = (min(acc_vals), max(acc_vals)) if acc_vals else (0.0, 1.0)
    y_min, y_max = gmin - 0.02 * abs(gmin), gmax + 0.02 * abs(gmax)

    # ---------------- colour base ----------------
    base_cmap   = plt.get_cmap('tab10')
    note_colour = {n: base_cmap(i % 10) for i, n in enumerate(notes)}

    # ---------------- figure ----------------
    right_margin = 0.28
    fig_w = max(4 * len(weights), 5)
    fig, axes = plt.subplots(1, len(weights), figsize=(fig_w, 5),
                             sharey=True, constrained_layout=False)
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    fig.subplots_adjust(right=1 - right_margin, wspace=0.25)

    side_handles = {}

    for ax, w in zip(axes, weights):
        # ---------- filter runs of this weight ----------
        runs_w = [r for r in runs_data if _get(r, weight_field) == w]
        if not runs_w:
            continue

        # ---------- build accuracy dict ----------
        # acc[model][(note, exp)] = value
        acc = {m: {} for m in models}
        for r in runs_w:
            m = _get(r, model_field)
            n = _get(r, notes_field)
            e = _get(r, exps_field)
            v = _get(r, accuracy_field)
            if v is not None:
                acc[m][(n, e)] = v

        # ---------- drop empty variants & collect per-note exp sets ----------
        note_to_exps = {n: set() for n in notes}
        for d in acc.values():
            for (n, e), v in d.items():
                note_to_exps[n].add(e)

        # remove notes with no data at all (all‐NaN already excluded above)
        note_to_exps = {n: sorted(es) for n, es in note_to_exps.items() if es}

        # ---------- uniform bar width ----------
        max_vars_per_model = max(len(d) for d in acc.values())
        bar_w = 0.8 / max_vars_per_model   # 0.8 is the model slot width

        # ---------- shade mapping (note → exp → colour) ----------
        shade_cache = {}
        for n, exps in note_to_exps.items():
            if len(exps) == 1:                   # single variant -> mid-shade
                shade_cache[n] = {exps[0]: shade_color(note_colour[n], 0.55)}
            else:
                fracs = np.linspace(0.25, 0.85, len(exps))
                shade_cache[n] = {e: shade_color(note_colour[n], f)
                                   for e, f in zip(exps, fracs)}

        # ---------- compute bar positions ----------
        x_ticks = []
        for i, m in enumerate(models):
            variants = sorted(acc[m].keys())     # deterministic order
            start = i - 0.4 + bar_w / 2          # left edge of model slot
            for k, var in enumerate(variants):
                n, e = var
                val  = acc[m][var]
                x_pos = start + k * bar_w

                # draw bar
                col = shade_cache[n][e]
                ax.bar(x_pos, val, width=bar_w, color=col, edgecolor='black')

                # value label
                ax.text(x_pos, val + 0.005 * abs(gmax), f"{val:.2f}",
                        ha='center', va='bottom', fontsize=6)

                # collect legend handle once
                lbl = f"{legend_replacer.get(n, n)} | {legend_replacer.get(e, e)}"
                if lbl not in side_handles:
                    side_handles[lbl] = mpatches.Patch(facecolor=col, label=lbl)

            x_ticks.append(i)

        # ---------- axis cosmetics ----------
        ax.set_title(f"{weight_field.split('.', 1)[1]} = {w}")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_xlabel(model_field.split('.', 1)[1])
        ax.set_ylim(y_min, y_max)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    if axes:
        axes[0].set_ylabel(accuracy_field.split('.', 1)[1])

    # ---------- top legend (notes) ----------
    top_handles = [
        mpatches.Patch(facecolor=note_colour[n],
                       label=legend_replacer.get(n, n))
        for n in note_to_exps
    ]
    if top_handles:
        axes[0].legend(handles=top_handles,
                       title=legend_replacer.get(notes_field.split('.', 1)[1],
                                                 notes_field.split('.', 1)[1]),
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1.23),
                       ncol=max(1, len(top_handles)),
                       frameon=False)

    # ---------- side legend (combined) ----------
    if side_handles:
        axes[-1].legend(handles=list(side_handles.values()),
                        title=(f"{legend_replacer.get(notes_field.split('.', 1)[1], notes_field.split('.', 1)[1])} × "
                               f"{legend_replacer.get(exps_field.split('.', 1)[1], exps_field.split('.', 1)[1])}"),
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=False)

    plt.show(block=True)




LEGEND_REPLACER_EXP = {
        # '001': 'baseline kl',
        '200': 'baseline kl v2',
        '100': 'baseline kl v2',
        '300': 'baseline kl v2',
        '301': 'baseline kl simd group',
        # '002': 'mul per ch, mean',
        # '003': 'mul per elm, mean',
        # '004': 'mul per elm bug, mean',
        # '005': 'mul per elm, sum',
        # '006': 'mul per elm, sum',
        # '007': 'mul per elm, sum',
        # '008': 'mul per elm, sum',
        # '009': 'simd with entropy grouping', # BUG! Problem with the ordering
        # '010': 'simd with hessian grouping', # BUG! Did not compute hessians- they where all ones
        # '011': '011',
        # '012': '012',
        # '013': 'simd with mse grouping (clustering)', # BUG! Problem with the ordering
        # '014': 'simd histogram grouping alg',
        # '015': 'simd group by hessians (all ones)', # BUG! Did not compute hessians- they were all ones (same results as 010)
        # '016': 'simd group by entropy',
        # '017': 'simd group by mse using k-means',
        # '018': 'simd group by hessians',
        # '019': 'simd random group iters',
        # '020': 'simd random group iters, more candidates',
        # '021': 'simd random group iters, more candidates, fixed bug',
    }

def main():
    # Define filters for runs (adjust as needed)
    filters = {
        "config.exp": {"$in": list(LEGEND_REPLACER_EXP.keys())},
        "config.weight_n_bits": {"$in": [
            3,
            # 4,
        ]},
        "config.mp_per_channel_cost": {"$in": [
            'KL',
            # 'SQNR',
            # 'HMSE_SUM',
            # 'HMSE_MEAN',
            # 'MSE',
            # 'EWQ'
        ]},
        "config.wandb_notes": {"$in": [
            'mp',
            # 'mp-ch',
            # 'mp-ch-simd16',
            'mp-ch-simd32',
            # 'mp-ch-EWQ',
            # 'mp-ch-hmse',
        ]},
        "config.model_name": {"$in": [
            'resnet18',
            'resnet50',
            'mobilenetv2',
            'deit_t',
            'vit_s',
            'deit_s',
            'vit_b',
            'deit_b',
            'swin_s',
            'swin_b',
        ]},
    }

    analyzer = WandBAnalyzer(ENTITY, PROJECT, cache_path=SAVE_PATH)

    # Set force_refresh=True if you want to ignore the cache and re-fetch
    runs_data = analyzer.fetch_runs(filters, force_refresh=True)

    # data_dict = analyzer.extract_data(runs_data, x_field, y_field, line_field)


if __name__ == "__main__":
    main()
    # Load cached runs_data
    with open(SAVE_PATH, "rb") as f:
        runs_data = pickle.load(f)

    plot_accuracy_bar_subplots(
        runs_data,
        weight_field="config.weight_n_bits",
        model_field="config.model_name",
        # notes_field="config.mp_per_channel_cost",
        notes_field="config.wandb_notes",
        exps_field="config.exp",
        accuracy_field="summary.compressed_accuracy",
        legend_replacer=LEGEND_REPLACER_EXP,
    )
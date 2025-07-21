import numpy as np
import pickle

import os

import wandb
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # added import for patch legend handles
plt.ion()  # turn on interactive mode

SAVE_PATH = '/Vols/vol_design/tools/swat/users/liord/temp/runs_cache.pkl'
legend_replacer = {'mp-ch-fine': 'mp per ch',
                   'mp-ch': 'mp per ch',
                   'mp-ch-EWQ': 'mp per ch with EWQ',
                   'mp-ch-hmse': 'mp per ch with hmse',
                   'low rank': 'MLoRQ',
                   'mp-ch-simd32-fine': 'mp per ch simd32',
                   'mp-ch-simd16-fine': 'mp per ch simd16',
                   'mp-fine': 'regular mp'}
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


def plot_accuracy_bar_subplots(
    runs_data: list,
    weight_field: str,
    model_field: str,
    notes_field: str,
    exps_field: str,
    accuracy_field: str
) -> None:
    """
    Subplots per weight; within each:
      - Bars grouped by wandb_notes (color).
      - Within each group, bars per exp_notes differ by hatch pattern.
      - Annotate each bar with accuracy.
      - Legends for both colors (wandb_notes) and patterns (exp_notes).
    """
    def _get(run, field):
        prefix, key = field.split(".", 1)
        return run[prefix].get(key)

    weights = sorted({ _get(r, weight_field) for r in runs_data })
    models  = sorted({ _get(r, model_field)  for r in runs_data })
    notes   = sorted({ _get(r, notes_field)  for r in runs_data })
    exps    = sorted({ _get(r, exps_field)   for r in runs_data })

    accs = [ _get(r, accuracy_field) for r in runs_data if _get(r, accuracy_field) is not None ]
    gmin, gmax = (min(accs), max(accs)) if accs else (0.0, 1.0)
    y_min = gmin - 0.02 * abs(gmin)
    y_max = gmax + 0.02 * abs(gmax)

    cmap = plt.get_cmap("tab10")
    color_map = { note: cmap(i % 10) for i, note in enumerate(notes) }
    patterns = ['/', '\\', 'x', '-', '+', 'o', '.', '*']
    hatch_map = { exp: patterns[i % len(patterns)] for i, exp in enumerate(exps) }

    fig, axes = plt.subplots(1, len(weights), figsize=(4 * len(weights), 5), sharey=True)
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    for ax, w in zip(axes, weights):
        runs_w = [r for r in runs_data if _get(r, weight_field) == w]
        x = np.arange(len(models))
        total_w = 0.8
        cluster_w = total_w / len(notes)
        bar_w = cluster_w / len(exps)

        acc = np.full((len(models), len(notes), len(exps)), np.nan)
        for r in runs_w:
            i = models.index(_get(r, model_field))
            j = notes.index(_get(r, notes_field))
            k = exps.index(_get(r, exps_field))
            val = _get(r, accuracy_field)
            if val is not None:
                acc[i, j, k] = val

        for j, note in enumerate(notes):
            center = x - total_w/2 + (j + 0.5) * cluster_w
            for k, exp in enumerate(exps):
                offs = center - cluster_w/2 + (k + 0.5) * bar_w
                heights = acc[:, j, k]
                bars = ax.bar(
                    offs,
                    heights,
                    width=bar_w,
                    color=color_map[note],
                    hatch=hatch_map[exp],
                    edgecolor="black"
                )
                for bar, h in zip(bars, heights):
                    if not np.isnan(h):
                        ax.text(
                            bar.get_x() + bar.get_width()/2,
                            h + 0.005 * abs(gmax),
                            f"{h:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=6
                        )

        ax.set_title(f"{weight_field.split('.',1)[1]} = {w}")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_xlabel(model_field.split(".",1)[1])
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel(accuracy_field.split(".",1)[1])

    color_handles = [mpatches.Patch(facecolor=color_map[n], label=n) for n in notes]
    pattern_handles = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_map[e], label=e)
        for e in exps
    ]

    fig.legend(
        handles=color_handles + pattern_handles,
        title="Legend",
        ncol=len(notes) + len(exps),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=True)


def main():
    """
    Example usage of WandBAnalyzer with caching.
    """
    # Replace with your W&B entity and project names
    ENTITY = "sony-semi-ml"
    PROJECT = "MpPerCh_May2025"

    # Define filters for runs (adjust as needed)
    filters = {
        "config.exp": {"$in": ['028', '029', '030', '034', '035', '036', '037', '038', '039', '040'
                               '041','042', '043', '044']},
        "config.exp_notes": {"$in": ['KL, 2-4-8, BOAStar', 'KL, 2-4-8, 200 iter', 'KL, 2-4-8, GREEDY',
                                     'KL, 2-4-8, lambda_frontier_candidates']},
        # "config.wandb_notes": {"$nin": ['fixed-fine']},
        "config.weight_n_bits": {"$in": [3]},
        # "config.wandb_notes": {"$in": ['mp-ch-fine', 'mp-ch-simd32-fine', 'mp-ch-simd16-fine', 'mp-fine', 'low rank']},
        # "config.wandb_notes": {"$in": ['mp-ch-fine', 'mp-ch-simd32-fine', 'mp-ch-simd16-fine', 'mp-fine']},
        "config.wandb_notes": {"$in": ['mp-ch', 'mp-ch-EWQ', 'mp-ch-hmse']},
        # "config.model_name": {"$in": ['resnet18', 'resnet50', 'mobilenetv2', 'deit_t', 'vit_s', 'deit_s', 'vit_b', 'deit_b', 'swin_s', 'swin_b']},
        # "config.model_name": {"$in": ['resnet18', 'resnet50', 'mobilenetv2']},
        # "config.model_name": {"$in": ['deit_t', 'deit_s', 'deit_b', 'vit_s', 'vit_b', 'swin_s', 'swin_b']},
        "config.model_name": {"$in": ['deit_t']},
    }

    # Fields for plotting (must be prefixed with "config." or "summary.")
    x_field = "config.weight_n_bits"
    y_field = "summary.compressed_accuracy"
    # y_field = "summary.acc_before_finetune"
    line_field = "config.wandb_notes"  # or "summary.wandb_notes"

    analyzer = WandBAnalyzer(ENTITY, PROJECT, cache_path=SAVE_PATH)

    # Set force_refresh=True if you want to ignore the cache and re-fetch
    runs_data = analyzer.fetch_runs(filters, force_refresh=True)
    # runs_data = analyzer.fetch_runs(filters, force_refresh=False)

    data_dict = analyzer.extract_data(runs_data, x_field, y_field, line_field)
    # analyzer.plot(
    #     data_dict,
    #     x_label="weight_n_bits",
    #     y_label="accuracy",
    #     title="Accuracy vs Weight Bits"
    # )


if __name__ == "__main__":
    main()
    # Load cached runs_data
    with open(SAVE_PATH, "rb") as f:
        runs_data = pickle.load(f)

    plot_accuracy_bar_subplots(
        runs_data,
        weight_field="config.weight_n_bits",
        model_field="config.model_name",
        notes_field="config.wandb_notes",
        exps_field="config.exp_notes",
        accuracy_field="summary.compressed_accuracy"
        # accuracy_field="summary.acc_before_finetune"
    )
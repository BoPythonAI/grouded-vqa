from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#8C8C8C"
DARK = "#333333"
LIGHT_GRAY = "#D7D7D7"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_rows(filename: str) -> list[dict[str, str]]:
    with (REPORTS / filename).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def export_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        kwargs = {"dpi": 220} if extension == "png" else {}
        fig.savefig(FIGURES / f"{stem}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def dumbbell(
    ax: plt.Axes,
    labels: list[str],
    baseline: list[float],
    adapted: list[float],
    xlim: tuple[float, float],
    first_label: str = "InstructBLIP zero-shot",
    second_label: str = "E6 LLM LoRA",
) -> None:
    y = np.arange(len(labels))
    for y_value, first, second in zip(y, baseline, adapted, strict=True):
        ax.plot([first, second], [y_value, y_value], color=LIGHT_GRAY, linewidth=2.0, zorder=1)
        ax.annotate(
            f"{first:.1f}",
            (first, y_value),
            xytext=(-4, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color=DARK,
        )
        ax.annotate(
            f"{second:.1f}",
            (second, y_value),
            xytext=(4, -7),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7,
            color=BLUE,
        )
    ax.scatter(baseline, y, color=GRAY, marker="s", s=35, label=first_label, zorder=3)
    ax.scatter(adapted, y, color=BLUE, marker="o", s=38, label=second_label, zorder=4)
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    ax.invert_yaxis()
    ax.grid(axis="x")


def figure_core_vqa() -> None:
    rows = read_rows("core_vqa_results.csv")
    order = ["E0", "E1", "E2", "E4", "E3", "E3b", "E5", "E6"]
    by_run = {row["run"]: row for row in rows}
    selected = [by_run[run] for run in order]
    labels = [
        "E0  BLIP-2 zero-shot",
        "E1  BLIP-2 Q-Former LoRA",
        "E2  BLIP-2 LLM LoRA",
        "E4  BLIP-2 dual LoRA",
        "E3  InstructBLIP default",
        "E3b InstructBLIP short prompt",
        "E5  InstructBLIP LLM LoRA (1k)",
        "E6  InstructBLIP LLM LoRA (10k)",
    ]
    overall = np.array([float(row["overall"]) for row in selected])
    categories = np.array(
        [[float(row[column]) for column in ("number", "other", "yes_no")] for row in selected]
    )
    colors = [GRAY, GRAY, GRAY, GRAY, SKY, SKY, ORANGE, BLUE]

    fig, (ax_bar, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(10.4, 5.1),
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.24},
    )
    y = np.arange(len(labels))
    ax_bar.barh(y, overall - 55, left=55, color=colors, height=0.64)
    for y_value, value in zip(y, overall, strict=True):
        ax_bar.text(value + 0.25, y_value, f"{value:.2f}", va="center", fontsize=8)
    ax_bar.set_yticks(y, labels)
    ax_bar.set_xlim(55, 73)
    ax_bar.set_xticks([55, 60, 65, 70])
    ax_bar.set_xlabel("Overall VQAv2 score (%)")
    ax_bar.set_title("Overall performance")
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x")
    add_panel_label(ax_bar, "a")

    image = ax_heat.imshow(categories, cmap="viridis", vmin=40, vmax=90, aspect="auto")
    for row_index in range(categories.shape[0]):
        for column_index in range(categories.shape[1]):
            value = categories[row_index, column_index]
            ax_heat.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if value < 68 else "#111111",
                fontsize=8,
            )
    ax_heat.set_xticks(range(3), ["Number", "Other", "Yes/No"])
    ax_heat.set_yticks([])
    ax_heat.set_title("Answer-type scores (%)")
    ax_heat.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)
    colorbar.set_label("VQAv2 score (%)")
    add_panel_label(ax_heat, "b")
    fig.text(0.5, 0.01, "All rows use the same fixed 1,000-example validation subset.", ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.10)
    export_figure(fig, "core_vqa_progress")


def figure_full_validation() -> None:
    rows = read_rows("core_vqa_results.csv")
    by_run = {row["run"]: row for row in rows}
    direct = by_run["E15"]
    reranked = by_run["E19"]
    metrics = ["Overall", "Number", "Other", "Yes/No"]
    columns = ["overall", "number", "other", "yes_no"]
    first = [float(direct[column]) for column in columns]
    second = [float(reranked[column]) for column in columns]

    fig, (ax_scores, ax_invalid) = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.6),
        gridspec_kw={"width_ratios": [1.65, 0.8], "wspace": 0.36},
    )
    dumbbell(
        ax_scores,
        metrics,
        first,
        second,
        (48, 93),
        first_label="E15 direct",
        second_label="E19 reranked",
    )
    ax_scores.set_xlabel("Full VQAv2 validation score (%)")
    ax_scores.set_title("Direct decoding vs. short-answer reranking")
    ax_scores.legend(frameon=False, loc="upper right")
    add_panel_label(ax_scores, "a")

    invalid = [float(direct["invalid_yes_no"]), float(reranked["invalid_yes_no"])]
    labels = ["E15 direct", "E19 reranked"]
    bars = ax_invalid.barh(labels, invalid, color=[GRAY, BLUE], height=0.55)
    for bar, value in zip(bars, invalid, strict=True):
        ax_invalid.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    ax_invalid.set_xlim(0, 4.0)
    ax_invalid.set_xlabel("Invalid yes/no outputs (%) ↓")
    ax_invalid.set_title("Format failures")
    ax_invalid.invert_yaxis()
    ax_invalid.grid(axis="x")
    add_panel_label(ax_invalid, "b")
    fig.text(0.5, -0.02, "Both runs evaluate all 214,354 VQAv2 validation questions.", ha="center", fontsize=8)
    export_figure(fig, "full_validation_decoding")


def figure_hallucination() -> None:
    rows = read_rows("hallucination_results.csv")
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["benchmark"], {})[row["model"]] = row
    baseline_name = "InstructBLIP zero-shot"
    adapted_name = "E6 LLM LoRA r8"

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), gridspec_kw={"wspace": 0.48})
    panels = [
        (
            "POPE · 9,000 questions",
            "POPE",
            [("Accuracy ↑", "accuracy"), ("Precision ↑", "precision"), ("Recall ↑", "recall"), ("F1 ↑", "f1")],
        ),
        (
            "CHAIR · 500 images",
            "CHAIR",
            [("CHAIRs ↓", "chair_s"), ("CHAIRi ↓", "chair_i"), ("Object recall ↑", "object_recall")],
        ),
        (
            "HallusionBench · 1,129 questions",
            "HallusionBench",
            [
                ("Question ↑", "accuracy"),
                ("Question-pair ↑", "question_pair"),
                ("Figure ↑", "figure"),
                ("Hard pair ↑", "hard_pair"),
            ],
        ),
    ]
    for index, (title, benchmark, metrics) in enumerate(panels):
        labels = [label for label, _ in metrics]
        baseline = [float(grouped[benchmark][baseline_name][column]) for _, column in metrics]
        adapted = [float(grouped[benchmark][adapted_name][column]) for _, column in metrics]
        dumbbell(axes[index], labels, baseline, adapted, (0, 100))
        axes[index].set_xlabel("Score / rate (%)")
        axes[index].set_title(title)
        axes[index].set_xticks([0, 25, 50, 75, 100])
        add_panel_label(axes[index], chr(ord("a") + index))
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GRAY, markeredgecolor=GRAY, label="InstructBLIP zero-shot"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, label="E6 LLM LoRA"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.20)
    export_figure(fig, "hallucination_benchmarks")


def figure_grounding_tradeoff() -> None:
    rows = read_rows("grounding_results.csv")
    by_run = {row["run"]: row for row in rows}
    exploratory_runs = ["E6", "E12a", "E12b", "E13a", "E13b", "E13c"]
    controls = {"E6", "E12a", "E13a"}

    fig, (ax_explore, ax_multiseed) = plt.subplots(1, 2, figsize=(9.8, 4.2), gridspec_kw={"wspace": 0.28})
    for run in exploratory_runs:
        row = by_run[run]
        x = float(row["vqa_overall"])
        y = float(row["grounding_overall"])
        is_control = run in controls
        ax_explore.scatter(
            x,
            y,
            color=GRAY if is_control else BLUE,
            marker="s" if is_control else "o",
            s=55,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        offsets = {
            "E6": (-27, -12),
            "E12a": (-28, 7),
            "E12b": (6, 5),
            "E13a": (5, -13),
            "E13b": (5, 5),
            "E13c": (-27, -12),
        }
        ax_explore.annotate(run, (x, y), xytext=offsets[run], textcoords="offset points", fontsize=8)
    ax_explore.set_xlim(70.55, 71.55)
    ax_explore.set_ylim(85.15, 87.05)
    ax_explore.set_xlabel("VQAv2 score (%) ↑")
    ax_explore.set_ylabel("Grounding score (%) ↑")
    ax_explore.set_title("Single-seed exploratory runs")
    ax_explore.grid()
    add_panel_label(ax_explore, "a")

    control_rows = [row for row in rows if row["run"].startswith("E14a-")]
    grounding_rows = [row for row in rows if row["run"].startswith("E14b-")]
    for label, selected, color, marker in (
        ("Control", control_rows, GRAY, "s"),
        ("+ grounding", grounding_rows, BLUE, "o"),
    ):
        vqa = np.array([float(row["vqa_overall"]) for row in selected])
        grounding = np.array([float(row["grounding_overall"]) for row in selected])
        ax_multiseed.errorbar(
            vqa.mean(),
            grounding.mean(),
            xerr=vqa.std(ddof=1),
            yerr=grounding.std(ddof=1),
            fmt=marker,
            color=color,
            markersize=8,
            capsize=3,
            elinewidth=1.2,
            label=label,
        )
        ax_multiseed.annotate(
            f"{vqa.mean():.3f}, {grounding.mean():.3f}",
            (vqa.mean(), grounding.mean()),
            xytext=(7, 7 if label == "Control" else -14),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    ax_multiseed.set_xlim(71.25, 71.38)
    ax_multiseed.set_ylim(85.65, 86.13)
    ax_multiseed.set_xlabel("VQAv2 score (%) ↑")
    ax_multiseed.set_ylabel("Grounding score (%) ↑")
    ax_multiseed.set_title("E14 mean ± SD across three seeds")
    ax_multiseed.grid()
    ax_multiseed.legend(frameon=False, loc="lower left")
    add_panel_label(ax_multiseed, "b")
    fig.text(0.5, 0.01, "Focused axes expose small trade-offs; every point is numerically labelled.", ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.16)
    export_figure(fig, "grounding_tradeoff")


def figure_alignment_ablation() -> None:
    rows = read_rows("alignment_ablation_results.csv")
    by_run = {row["run"]: row for row in rows}
    baseline = by_run["E6"]
    runs = ["E8a", "E8b", "E9a", "E9b", "E10a", "E10b", "E11a"]
    labels = {run: run for run in runs}
    labels["E11a"] = "E11a/b"
    objective_runs = {"E8b", "E9b", "E10b"}

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), gridspec_kw={"wspace": 0.26})
    comparisons = [
        ("mismatch_drop", "Δ mismatch accuracy drop (pp) ↑"),
        ("mismatch_answer_change", "Δ mismatch answer-change rate (pp) ↑"),
    ]
    offsets = {
        "E8a": (5, 5),
        "E8b": (5, -12),
        "E9a": (-25, 5),
        "E9b": (5, 5),
        "E10a": (5, 5),
        "E10b": (5, -12),
        "E11a": (5, 5),
    }
    answer_change_offsets = {
        **offsets,
        "E9b": (7, -14),
        "E11a": (-50, 5),
    }
    for ax, (column, ylabel) in zip(axes, comparisons, strict=True):
        ax.fill_between([0, 0.25], 0, 1.0, color="#E8F5EE", alpha=0.75, zorder=0)
        ax.axvline(0, color=DARK, linewidth=0.9)
        ax.axhline(0, color=DARK, linewidth=0.9)
        ax.scatter(0, 0, color=DARK, marker="*", s=90, label="E6 reference", zorder=5)
        for run in runs:
            row = by_run[run]
            x = float(row["vqa_overall"]) - float(baseline["vqa_overall"])
            y = float(row[column]) - float(baseline[column])
            is_objective = run in objective_runs
            ax.scatter(
                x,
                y,
                color=BLUE if is_objective else GRAY,
                marker="o" if is_objective else "s",
                s=48,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            label_offsets = answer_change_offsets if column == "mismatch_answer_change" else offsets
            ax.annotate(
                labels[run],
                (x, y),
                xytext=label_offsets[run],
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlim(-0.62, 0.25)
        ax.set_xlabel("Δ VQAv2 score vs. E6 (pp) ↑")
        ax.set_ylabel(ylabel)
        ax.grid()
    axes[0].set_ylim(-1.2, 0.2)
    axes[0].set_title("Accuracy-drop diagnostic")
    axes[1].set_ylim(-3.2, 1.0)
    axes[1].set_title("Answer-change diagnostic")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=DARK, markeredgecolor=DARK, markersize=10, label="E6 reference"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GRAY, markeredgecolor=GRAY, label="Matched control"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, label="Alignment objective"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.20)
    export_figure(fig, "alignment_ablation")


def main() -> None:
    configure_style()
    figure_core_vqa()
    figure_full_validation()
    figure_hallucination()
    figure_grounding_tradeoff()
    figure_alignment_ablation()
    print(f"Wrote README figures to {FIGURES}")


if __name__ == "__main__":
    main()

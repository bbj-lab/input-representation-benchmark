#!/usr/bin/env python3
"""Render family-averaged outcome x factor traces.

Leaves 08_outcome_factor_traces untouched. Each binary or regression panel
shows four clinical-family lines: the equal-weight mean of that family's
outcomes after the usual 3-seed × 2-architecture average.

Also writes a sibling set under 09_outcome_family_traces/baseline_delta/
with the same layout but y-values as Δ from each condition's baseline
factor level.

It reads the already written 24–48h metric tables; it does not rerun any
Stage 3 prediction scoring.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_LONG = (
    REPO_ROOT / "outputs/runs/ml4h_2026/metrics/metrics_long.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT.parent / "Archive/ml4h-dump/08_ML4H_provisional"
)
FAMILY_SECTION = "09_outcome_family_traces"
BASELINE_DELTA_SUBDIR = "baseline_delta"
EXPERIMENTS = ("exp1", "exp2", "exp3")


def submission_trace_stem(experiment: str) -> str:
    return f"{experiment}-trace-plot"
FAMILY_CELL_WIDTH_IN = 9.2
FAMILY_GUTTER_WIDTH_IN = 0.85
FAMILY_MARKER_SCALE = 1.7
FAMILY_SHAPE_LEGEND_SIZE = 12.5


def load_traces():
    script = Path(__file__).resolve().parent / "generate_ml4h_outcome_factor_traces.py"
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    import generate_ml4h_outcome_factor_traces as traces

    return traces


def family_members(
    source: pd.DataFrame,
    task_type: str,
    family: str,
    *,
    spider,
    traces,
) -> tuple[str, ...]:
    """Registry min/max pairing, keeping only outcomes present in this experiment."""
    available = set(source.loc[source["task_type"] == task_type, "outcome"].astype(str))
    return tuple(
        outcome
        for outcome in spider.outcome_order(task_type, available)
        if traces.OUTCOME_FAMILIES.get(outcome) == family
    )


def six_model_family_range(model_source: pd.DataFrame, *, traces) -> pd.DataFrame:
    """Min/max of the equal-weight family mean across the six trained models."""
    models = model_source.copy()
    models["outcome_family"] = models["outcome"].map(traces.OUTCOME_FAMILIES)
    missing = models["outcome_family"].isna()
    if missing.any():
        unknown = sorted(set(models.loc[missing, "outcome"].astype(str)))
        raise ValueError(f"Outcomes missing family mapping: {unknown}")
    family = models.groupby(
        [
            "experiment",
            "backbone",
            "seed",
            "condition",
            "level_id",
            "task_type",
            "metric",
            "outcome_family",
        ],
        as_index=False,
        sort=False,
    ).agg(family_point=("point", "mean"))
    ranges = family.groupby(
        ["experiment", "condition", "level_id", "task_type", "metric", "outcome_family"],
        as_index=False,
        sort=False,
    ).agg(
        lo=("family_point", "min"),
        hi=("family_point", "max"),
        n_models=("family_point", "size"),
    )
    if not (ranges["n_models"] == 6).all():
        bad = ranges.loc[ranges["n_models"] != 6]
        raise ValueError(f"Expected 6 models per family cell; found {bad.to_dict('records')}")
    return ranges


def average_over_families(
    source: pd.DataFrame,
    *,
    traces,
    spider,
    model_source: pd.DataFrame,
) -> pd.DataFrame:
    """Equal-weight mean of outcomes inside each clinical family."""
    group_keys = [
        "experiment",
        "backbone",
        "condition",
        "condition_title",
        "task_type",
        "metric",
        "level_id",
        "level_label",
        "level_position",
        "level_marker",
        "level_fillstyle",
        "is_baseline",
        "outcome_family",
        "outcome_color",
    ]
    grouped = source.groupby(group_keys, as_index=False, sort=False).agg(
        point=("point", "mean"),
        n_outcomes=("outcome", "nunique"),
        n_seeds=("n_seeds", "first"),
        n_architectures=("n_architectures", "first"),
        member_outcomes=("outcome", lambda values: "|".join(sorted(set(map(str, values))))),
    )
    ranges = six_model_family_range(model_source, traces=traces)
    merge_keys = [
        "experiment",
        "condition",
        "level_id",
        "task_type",
        "metric",
        "outcome_family",
    ]
    grouped = grouped.merge(ranges[merge_keys + ["lo", "hi", "n_models"]], on=merge_keys, how="left")
    if grouped["lo"].isna().any() or grouped["hi"].isna().any():
        raise ValueError("Missing six-model family range for at least one cell.")
    if grouped.empty:
        raise ValueError("No rows left after averaging families.")
    expected = {
        task_type: len(traces.FAMILY_ORDER[task_type]) for task_type in traces.TASK_ORDER
    }
    observed = grouped.groupby("task_type")["outcome_family"].nunique().to_dict()
    if observed != expected:
        raise ValueError(f"Expected four families per task; found {observed}.")
    member_lookup: dict[tuple[str, str], str] = {}
    for task_type in traces.TASK_ORDER:
        for family in traces.FAMILY_ORDER[task_type]:
            members = family_members(
                source,
                task_type,
                family,
                spider=spider,
                traces=traces,
            )
            member_lookup[(task_type, family)] = "|".join(members)
    grouped["outcome"] = grouped["outcome_family"]
    grouped["outcome_label"] = grouped["outcome_family"]
    grouped["outcome_linestyle"] = "solid"
    grouped["member_outcomes"] = [
        member_lookup[(task, family)]
        for task, family in zip(
            grouped["task_type"].astype(str), grouped["outcome_family"].astype(str)
        )
    ]
    panel = grouped.groupby(["condition", "task_type", "outcome"])["point"]
    panel_max = panel.transform("max")
    panel_min = panel.transform("min")
    varies = panel_max > panel_min
    grouped["level_delta"] = panel_max - panel_min
    grouped["is_best"] = varies & (grouped["point"] == panel_max)
    grouped["is_worst"] = varies & (grouped["point"] == panel_min)
    return grouped


def to_baseline_delta(
    source: pd.DataFrame,
    *,
    model_source: pd.DataFrame,
    traces,
) -> pd.DataFrame:
    """Replace family means with Δ relative to that condition's baseline level."""
    out = source.copy()
    keys = ["condition", "task_type", "outcome"]
    baseline = out.loc[out["is_baseline"], [*keys, "point"]].copy()
    if baseline.empty:
        raise ValueError("No baseline rows found for family-mean source.")
    if baseline.duplicated(keys).any():
        raise ValueError("Expected one baseline row per condition × task × family.")
    baseline = baseline.set_index(keys)["point"]
    index = pd.MultiIndex.from_frame(out[keys])
    out["baseline_point"] = baseline.reindex(index).to_numpy(dtype=float)
    if out["baseline_point"].isna().any():
        missing = (
            out.loc[out["baseline_point"].isna(), keys]
            .drop_duplicates()
            .astype(str)
            .to_dict(orient="records")
        )
        raise ValueError(f"Missing baseline family means for: {missing}")
    out["raw_point"] = out["point"].astype(float)
    out["point"] = out["raw_point"] - out["baseline_point"]
    models = model_source.copy()
    models["outcome_family"] = models["outcome"].map(traces.OUTCOME_FAMILIES)
    family = models.groupby(
        [
            "experiment",
            "backbone",
            "seed",
            "condition",
            "level_id",
            "task_type",
            "metric",
            "outcome_family",
        ],
        as_index=False,
        sort=False,
    ).agg(family_point=("point", "mean"))
    baseline_levels = (
        out.loc[out["is_baseline"], ["condition", "level_id"]]
        .drop_duplicates()
        .rename(columns={"level_id": "baseline_level_id"})
    )
    family = family.merge(baseline_levels, on="condition", how="left")
    base = family.loc[
        family["level_id"] == family["baseline_level_id"],
        [
            "experiment",
            "backbone",
            "seed",
            "condition",
            "task_type",
            "metric",
            "outcome_family",
            "family_point",
        ],
    ].rename(columns={"family_point": "baseline_family_point"})
    family = family.merge(
        base,
        on=[
            "experiment",
            "backbone",
            "seed",
            "condition",
            "task_type",
            "metric",
            "outcome_family",
        ],
        how="left",
    )
    if family["baseline_family_point"].isna().any():
        raise ValueError("Missing per-model baseline family mean.")
    family["delta"] = family["family_point"] - family["baseline_family_point"]
    delta_range = family.groupby(
        ["experiment", "condition", "level_id", "task_type", "metric", "outcome_family"],
        as_index=False,
        sort=False,
    ).agg(lo=("delta", "min"), hi=("delta", "max"))
    out = out.drop(columns=["lo", "hi"])
    out = out.merge(
        delta_range,
        on=["experiment", "condition", "level_id", "task_type", "metric", "outcome_family"],
        how="left",
    )
    if out["lo"].isna().any() or out["hi"].isna().any():
        raise ValueError("Missing six-model family-delta range.")
    panel = out.groupby(keys)["point"]
    panel_max = panel.transform("max")
    panel_min = panel.transform("min")
    varies = panel_max > panel_min
    out["level_delta"] = panel_max - panel_min
    out["is_best"] = varies & (out["point"] == panel_max)
    out["is_worst"] = varies & (out["point"] == panel_min)
    return out


def family_width_ratios(n_cond: int) -> list[float]:
    """Multi-condition: one column per condition. One-condition: AUROC | Spearman."""
    panel_count = 2 if n_cond == 1 else n_cond
    ratios: list[float] = []
    for panel_index in range(panel_count):
        ratios.append(FAMILY_CELL_WIDTH_IN)
        if panel_index < panel_count - 1:
            ratios.append(FAMILY_GUTTER_WIDTH_IN)
    return ratios


def place_family_shape_legend(ax, levels, *, title: str, traces, spider) -> None:
    ax.set_axis_off()
    ncol = 2 if len(levels) >= 2 else 1
    handles = []
    for level in levels:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=str(level["marker"]),
                color=traces.MARKER_EDGE_COLOR,
                linestyle="None",
                markerfacecolor=traces.SHAPE_LEGEND_FACE,
                markeredgecolor=traces.MARKER_EDGE_COLOR,
                markeredgewidth=traces.MARKER_EDGE_WIDTH,
                linewidth=0,
                markersize=FAMILY_SHAPE_LEGEND_SIZE,
                label=traces.wrap_level_label(str(level["label"])),
            )
        )
    legend = ax.legend(
        handles=traces.column_major(handles, ncol),
        title=title,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=ncol,
        frameon=False,
        fontsize=spider.LEGEND_PT,
        title_fontsize=spider.COLUMN_TITLE_PT,
        handlelength=2.0,
        columnspacing=1.25,
        handletextpad=0.40,
        labelspacing=0.40,
        borderaxespad=0.0,
        borderpad=0.20,
        prop={"family": "serif", "weight": "normal", "size": spider.LEGEND_PT},
    )
    legend._legend_box.align = "center"
    legend.get_title().set_fontfamily("serif")
    legend.get_title().set_ha("center")
    legend.get_title().set_color(spider.TEXT_COLOR)
    ax.add_artist(legend)


def draw_family_block(ax, *, family: str, members: list[str], color: str, spider) -> None:
    """One fixed cell: family line plus its member list, top-aligned."""
    ax.set_axis_off()
    ax.plot(
        [0.00, 0.10],
        [0.92, 0.92],
        color=color,
        linewidth=2.8,
        solid_capstyle="round",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.text(
        0.13,
        0.92,
        family,
        ha="left",
        va="center",
        fontsize=spider.OUTCOME_KEY_PT,
        fontweight="bold",
        color=spider.TEXT_COLOR,
        fontfamily="serif",
        transform=ax.transAxes,
        clip_on=False,
    )
    count = len(members)
    top = 0.76
    bottom = 0.05
    if count <= 1:
        positions = [top]
    else:
        # Keep room for eight Biomarker labels without clipping the cell.
        step = min(0.15, (top - bottom) / (count - 1))
        positions = [top - step * index for index in range(count)]
    for position, member in zip(positions, members):
        ax.text(
            0.13,
            position,
            member,
            ha="left",
            va="center",
            fontsize=spider.OUTCOME_KEY_PT,
            color=spider.TEXT_COLOR,
            fontfamily="serif",
            transform=ax.transAxes,
            clip_on=False,
        )


def family_key_styles(source: pd.DataFrame) -> pd.DataFrame:
    return (
        source.loc[
            :,
            ["task_type", "outcome", "outcome_color", "member_outcomes"],
        ]
        .drop_duplicates()
        .set_index(["task_type", "outcome"])
    )


def family_key_member_ids(styles: pd.DataFrame, task_type: str, family: str) -> list[str]:
    if (task_type, family) not in styles.index:
        return []
    return [
        outcome
        for outcome in str(styles.loc[(task_type, family), "member_outcomes"]).split("|")
        if outcome
    ]


def family_key_grid_shape(*, traces) -> tuple[int, int]:
    """Rows × columns of family blocks under each binary/regression title."""
    family_count = len(traces.FAMILY_ORDER[traces.TASK_ORDER[0]])
    n_cols = 2
    n_rows = (family_count + n_cols - 1) // n_cols
    return n_rows, n_cols


def family_key_row_member_counts(source: pd.DataFrame, *, traces) -> list[int]:
    """Max member count per key row across both task panels' 2×2 family grids."""
    styles = family_key_styles(source)
    n_rows, n_cols = family_key_grid_shape(traces=traces)
    row_member_counts: list[int] = []
    for row in range(n_rows):
        counts = []
        for task_type in traces.TASK_ORDER:
            families = traces.FAMILY_ORDER[task_type]
            for col in range(n_cols):
                family_index = row * n_cols + col
                if family_index >= len(families):
                    continue
                counts.append(
                    len(family_key_member_ids(styles, task_type, families[family_index]))
                )
        row_member_counts.append(max(counts) if counts else 1)
    return row_member_counts


def family_key_height_ratios(row_member_counts: list[int]) -> list[float]:
    # Biomarkers (8) shares the top row with Electrolytes, so that row stays taller.
    return [max(2.5, 0.32 * count) for count in row_member_counts]


def family_key_height_in(source: pd.DataFrame, *, traces) -> float:
    """Tall enough that the densest family row (Biomarkers) keeps readable spacing."""
    family_height_ratios = family_key_height_ratios(
        family_key_row_member_counts(source, traces=traces)
    )
    # Title band + two family rows + pad; 2×2 is shorter than the old 1×4 stack.
    return 1.15 + 0.55 + sum(family_height_ratios)


def place_family_key(fig, spec, source: pd.DataFrame, *, traces, spider) -> None:
    """Binary | regression panels, each with a 2×2 family block grid.

    The key band is shifted left of the plot y-label margin so the binary
    section does not sit in empty space under the AUROC axis label.
    """
    styles = family_key_styles(source)
    n_rows, n_cols = family_key_grid_shape(traces=traces)
    n_tasks = len(traces.TASK_ORDER)
    family_height_ratios = family_key_height_ratios(
        family_key_row_member_counts(source, traces=traces)
    )
    probe = fig.add_subplot(spec)
    x0, y0, width, height = probe.get_position().bounds
    probe.remove()
    key_left = min(x0, 0.025)
    key_right = x0 + width
    outer = fig.add_gridspec(
        1,
        1,
        left=key_left,
        right=key_right,
        bottom=y0,
        top=y0 + height,
    )
    key_grid = GridSpecFromSubplotSpec(
        1 + n_rows,
        n_tasks * n_cols,
        subplot_spec=outer[0, 0],
        height_ratios=[0.55, *family_height_ratios],
        hspace=0.22,
        wspace=0.16,
    )
    for task_index, task_type in enumerate(traces.TASK_ORDER):
        col0 = task_index * n_cols
        col1 = col0 + n_cols
        title = fig.add_subplot(key_grid[0, col0:col1])
        title.set_axis_off()
        title.text(
            0.00,
            0.40,
            traces.TASK_KEY_TITLES[task_type],
            ha="left",
            va="center",
            fontsize=spider.COLUMN_TITLE_PT,
            color=spider.TEXT_COLOR,
            fontfamily="serif",
            transform=title.transAxes,
            clip_on=False,
        )
        families = traces.FAMILY_ORDER[task_type]
        for family_index, family in enumerate(families):
            row = 1 + family_index // n_cols
            col = col0 + family_index % n_cols
            axis = fig.add_subplot(key_grid[row, col])
            members = [
                spider.FULL_OUTCOME_LABELS.get(outcome, outcome)
                for outcome in family_key_member_ids(styles, task_type, family)
            ]
            if not members:
                axis.set_axis_off()
                continue
            draw_family_block(
                axis,
                family=family,
                members=members,
                color=str(styles.loc[(task_type, family), "outcome_color"]),
                spider=spider,
            )


def plot_family_traces(
    *,
    experiment: str,
    source: pd.DataFrame,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    traces,
    spider,
    section: str = FAMILY_SECTION,
    stem: str | None = None,
    description: str | None = None,
    ylabel_lookup: dict[str, str] | None = None,
    reference_y: float | None = None,
) -> None:
    spider.style()
    columns = spider.panel_columns(experiment)
    layout = traces.level_layout(experiment, spider=spider)
    n_cond = len(columns)
    side_by_side_tasks = n_cond == 1
    width_ratios = family_width_ratios(n_cond)
    n_cols = len(width_ratios)
    width = sum(width_ratios)
    key_height = family_key_height_in(source, traces=traces)
    # Extra height for wrapped multi-level shape legends (Exp1 granularity).
    height_ratios = [2.55, traces.LETTER_ROW_HEIGHT_IN]
    if side_by_side_tasks:
        height_ratios.append(traces.TRACE_AXIS_HEIGHT_IN + traces.LABEL_ROW_EXTRA_IN)
    else:
        for task_index, _task_type in enumerate(traces.TASK_ORDER):
            extra = (
                traces.LABEL_ROW_EXTRA_IN
                if task_index == len(traces.TASK_ORDER) - 1
                else 0.0
            )
            height_ratios.append(traces.TRACE_AXIS_HEIGHT_IN + extra)
    height_ratios.extend([traces.KEY_GAP_HEIGHT_IN, key_height])
    # Baseline-delta y-labels ("Δ baseline …") need more left margin than raw scores.
    left_margin = 0.14 if ylabel_lookup is not None else 0.11
    fig = plt.figure(figsize=(width, sum(height_ratios)))
    grid = GridSpec(
        len(height_ratios),
        n_cols,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        left=left_margin,
        right=0.985,
        top=0.985,
        bottom=0.015,
        hspace=0.18,
        wspace=0.04,
    )
    y_limits = {}
    for task_type in traces.TASK_ORDER:
        subset = source.loc[source["task_type"] == task_type]
        values = list(subset["point"])
        if "lo" in subset.columns:
            values.extend(subset["lo"].tolist())
            values.extend(subset["hi"].tolist())
        y_limits[task_type] = traces.padded_limits(values)
    ylabel_map = ylabel_lookup or {}

    def draw_one(
        axis,
        *,
        rows: pd.DataFrame,
        levels,
        families,
        task_type: str,
        show_level_labels: bool,
        show_ylabel: bool,
    ) -> None:
        traces.draw_trace_panel(
            axis,
            rows,
            levels=levels,
            outcomes=families,
            task_type=task_type,
            show_level_labels=show_level_labels,
            show_ylabel=show_ylabel,
            spider=spider,
            ylim=y_limits[task_type],
            marker_scale=FAMILY_MARKER_SCALE,
            ylabel=ylabel_map.get(task_type),
            reference_y=reference_y,
            show_range=True,
        )

    if side_by_side_tasks:
        column = columns[0]
        place_family_shape_legend(
            fig.add_subplot(grid[0, :]),
            layout[column.key],
            title=column.title,
            traces=traces,
            spider=spider,
        )
        for task_index, _task_type in enumerate(traces.TASK_ORDER):
            traces.draw_panel_letter(
                fig.add_subplot(grid[1, traces.task_panel_col(task_index)]),
                spider.PANEL_LETTERS[task_index],
                spider=spider,
            )
        for task_index, task_type in enumerate(traces.TASK_ORDER):
            families = traces.FAMILY_ORDER[task_type]
            axis = fig.add_subplot(grid[2, traces.task_panel_col(task_index)])
            rows = source.loc[
                (source["condition"] == column.key) & (source["task_type"] == task_type)
            ]
            draw_one(
                axis,
                rows=rows,
                levels=layout[column.key],
                families=families,
                task_type=task_type,
                show_level_labels=True,
                show_ylabel=True,
            )
    else:
        for column_index, column in enumerate(columns):
            col = traces.plot_column(n_cond, column_index)
            place_family_shape_legend(
                fig.add_subplot(grid[0, col]),
                layout[column.key],
                title=column.title,
                traces=traces,
                spider=spider,
            )
            traces.draw_panel_letter(
                fig.add_subplot(grid[1, col]),
                spider.PANEL_LETTERS[column_index],
                spider=spider,
            )
        for task_index, task_type in enumerate(traces.TASK_ORDER):
            row = 2 + task_index
            families = traces.FAMILY_ORDER[task_type]
            for column_index, column in enumerate(columns):
                axis = fig.add_subplot(grid[row, traces.trace_panel_col(column_index)])
                rows = source.loc[
                    (source["condition"] == column.key)
                    & (source["task_type"] == task_type)
                ]
                draw_one(
                    axis,
                    rows=rows,
                    levels=layout[column.key],
                    families=families,
                    task_type=task_type,
                    show_level_labels=task_index == len(traces.TASK_ORDER) - 1,
                    show_ylabel=column_index == 0,
                )
    fig.add_subplot(grid[-2, :]).set_axis_off()
    place_family_key(fig, grid[-1, :], source, traces=traces, spider=spider)
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        stem=stem or f"{experiment}_outcome_family_traces",
        source=source,
        description=description
        or (
            f"{spider.EXPERIMENT_DISPLAY[experiment]} 24-48h family-mean profiles: "
            "equal-weight mean of outcomes inside each clinical family after the "
            "3-seed × 2-architecture average; whiskers span the six per-model family means."
        ),
        input_hashes=input_hashes,
        section=section,
    )


def plot_family_deltas(
    *,
    experiment: str,
    source: pd.DataFrame,
    summary: pd.DataFrame,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    traces,
    spider,
) -> None:
    spider.style()
    columns = spider.panel_columns(experiment)
    layout = traces.level_layout(experiment, spider=spider)
    n_cond = len(columns)
    side_by_side_tasks = n_cond == 1
    n_cols = 2 if side_by_side_tasks else n_cond
    height_ratios = [2.15, traces.LETTER_ROW_HEIGHT_IN]
    panel_height = max(2.8, 4 * 0.48)
    if side_by_side_tasks:
        height_ratios.append(panel_height)
    else:
        for _task_type in traces.TASK_ORDER:
            height_ratios.append(panel_height)
    x_limits = {
        task_type: traces.padded_limits(
            summary.loc[summary["task_type"] == task_type, "delta"],
            lower=0.0,
        )
        for task_type in traces.TASK_ORDER
    }
    width = FAMILY_CELL_WIDTH_IN * n_cols
    fig = plt.figure(figsize=(width, sum(height_ratios)))
    grid = GridSpec(
        len(height_ratios),
        n_cols,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=[FAMILY_CELL_WIDTH_IN] * n_cols,
        left=0.20,
        right=0.99,
        top=0.985,
        bottom=0.04,
        hspace=0.22,
        wspace=0.42,
    )
    if side_by_side_tasks:
        column = columns[0]
        place_family_shape_legend(
            fig.add_subplot(grid[0, :]),
            layout[column.key],
            title=column.title,
            traces=traces,
            spider=spider,
        )
        for task_index, task_type in enumerate(traces.TASK_ORDER):
            traces.draw_panel_letter(
                fig.add_subplot(grid[1, task_index]),
                spider.PANEL_LETTERS[task_index],
                spider=spider,
            )
            rows = summary.loc[
                (summary["condition"] == column.key)
                & (summary["task_type"] == task_type)
            ]
            traces.draw_delta_panel(
                fig.add_subplot(grid[2, task_index]),
                rows,
                task_type=task_type,
                spider=spider,
                xlim=x_limits[task_type],
                marker_scale=FAMILY_MARKER_SCALE,
            )
    else:
        for column_index, column in enumerate(columns):
            place_family_shape_legend(
                fig.add_subplot(grid[0, column_index]),
                layout[column.key],
                title=column.title,
                traces=traces,
                spider=spider,
            )
            traces.draw_panel_letter(
                fig.add_subplot(grid[1, column_index]),
                spider.PANEL_LETTERS[column_index],
                spider=spider,
            )
        for task_index, task_type in enumerate(traces.TASK_ORDER):
            row = 2 + task_index
            for column_index, column in enumerate(columns):
                rows = summary.loc[
                    (summary["condition"] == column.key)
                    & (summary["task_type"] == task_type)
                ]
                traces.draw_delta_panel(
                    fig.add_subplot(grid[row, column_index]),
                    rows,
                    task_type=task_type,
                    spider=spider,
                    xlim=x_limits[task_type],
                    marker_scale=FAMILY_MARKER_SCALE,
                )
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        stem=f"{experiment}_outcome_family_delta",
        source=summary,
        description=(
            f"{spider.EXPERIMENT_DISPLAY[experiment]} 24-48h best-minus-worst gap "
            "per clinical family after averaging outcomes, 3 seeds, and 2 architectures."
        ),
        input_hashes=input_hashes,
    )


def write_figure(
    *,
    output_root: Path,
    entries: list[dict[str, object]],
    figure: plt.Figure,
    stem: str,
    source: pd.DataFrame,
    description: str,
    input_hashes: dict[str, str],
    section: str = FAMILY_SECTION,
) -> None:
    traces = load_traces()
    section_dir = output_root / section
    sources_dir = output_root / "sources"
    metadata_dir = output_root / "metadata"
    for directory in (section_dir, sources_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    source_path = sources_dir / f"{stem}.csv"
    metadata_path = metadata_dir / f"{stem}.json"
    pdf_path = section_dir / f"{stem}.pdf"
    png_path = section_dir / f"{stem}.png"
    source.to_csv(source_path, index=False)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.10)
    figure.savefig(png_path, dpi=traces.FIGURE_PNG_DPI, bbox_inches="tight", pad_inches=0.10)
    plt.close(figure)
    metadata_path.write_text(
        json.dumps(
            {
                "analysis": "24to48h",
                "created_at": traces.utc_now(),
                "description": description,
                "section": section,
                "source": str(source_path.relative_to(output_root)),
                "source_sha256": traces.sha256(source_path),
                "input_hashes": input_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    entries.append(
        {
            "section": section,
            "stem": stem,
            "description": description,
            "pdf": str(pdf_path.relative_to(output_root)),
            "png": str(png_path.relative_to(output_root)),
            "source": str(source_path.relative_to(output_root)),
            "metadata": str(metadata_path.relative_to(output_root)),
        }
    )


def copy_submission_traces(
    *,
    output_root: Path,
    entries: list[dict[str, object]],
    destination: Path,
    experiments: tuple[str, ...],
) -> list[str]:
    """Copy the requested family-trace PDFs into the paper folder as expN-trace-plot."""
    destination = destination.expanduser().resolve()
    sources_dir = destination / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    wanted = {
        f"{experiment}_outcome_family_traces": submission_trace_stem(experiment)
        for experiment in experiments
    }
    for entry in entries:
        stem = str(entry["stem"])
        dest_stem = wanted.get(stem)
        if dest_stem is None:
            continue
        pdf_src = output_root / str(entry["pdf"])
        png_src = output_root / str(entry["png"])
        csv_src = output_root / str(entry["source"])
        shutil.copy2(pdf_src, destination / f"{dest_stem}.pdf")
        shutil.copy2(png_src, destination / f"{dest_stem}.png")
        shutil.copy2(csv_src, sources_dir / f"{dest_stem}.csv")
        copied.append(dest_stem)
    missing = set(wanted.values()) - set(copied)
    if missing:
        raise FileNotFoundError(f"Submission copy missing traces: {sorted(missing)}")
    return sorted(copied)


def write_section_manifest(
    *,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    section: str = FAMILY_SECTION,
    extra: dict[str, object] | None = None,
) -> None:
    traces = load_traces()
    section_dir = output_root / section
    section_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(entries).to_csv(section_dir / "figure_index.csv", index=False)
    payload: dict[str, object] = {
        "created_at": traces.utc_now(),
        "generator": str(Path(__file__).resolve()),
        "analysis": "24to48h",
        "section": section,
        "figure_count": len(entries),
        "figures": entries,
        "input_hashes": input_hashes,
    }
    if extra:
        payload.update(extra)
    (section_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_baseline_delta_metadata(
    *,
    output_root: Path,
    experiments: tuple[str, ...],
    spider,
) -> Path:
    section_dir = output_root / FAMILY_SECTION / BASELINE_DELTA_SUBDIR
    section_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Family baseline-delta traces",
        "",
        "Sibling of the raw family-mean traces in the parent",
        "`09_outcome_family_traces/` directory. Layout, colors, markers, and family",
        "key are unchanged. The y-axis is the difference from that condition's",
        "baseline factor level after the usual family mean of outcomes.",
        "",
        "Baseline rows sit at Δ = 0. A dashed guide marks zero on each panel.",
        "",
    ]
    for experiment in experiments:
        lines.extend(
            [
                f"## {spider.EXPERIMENT_DISPLAY[experiment]}",
                "",
                f"- **File stem:** `{experiment}_outcome_family_baseline_delta_traces`",
                f"- **Baseline:** {spider.BASELINE_DISPLAY[experiment]}",
                f"- **Horizon:** 24–48h follow-up only",
                "- **Y-axis:** Δ baseline AUROC (binary) and Δ baseline Spearman ρ "
                "(regression)",
                "",
            ]
        )
    path = section_dir / "FIGURE_METADATA.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate(
    *,
    output_root: Path,
    experiments: tuple[str, ...] = EXPERIMENTS,
    submission_figures_dir: Path | None = None,
    metrics_long: Path | None = None,
) -> dict[str, object]:
    if not experiments:
        raise ValueError("At least one experiment is required.")
    unknown = set(experiments) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unsupported experiments: {sorted(unknown)}")
    traces = load_traces()
    spider = traces.load_spider_renderer()
    alternatives = traces.load_alternatives_renderer()
    metrics_path = (
        metrics_long.expanduser().resolve()
        if metrics_long is not None
        else DEFAULT_METRICS_LONG.expanduser().resolve()
    )
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metric table: {metrics_path}")
    metrics = pd.read_parquet(metrics_path)
    script_path = Path(__file__).resolve()
    input_hashes = {
        str(metrics_path): traces.sha256(metrics_path),
        str(script_path): traces.sha256(script_path),
        str(Path(traces.__file__).resolve()): traces.sha256(Path(traces.__file__).resolve()),
    }
    entries: list[dict[str, object]] = []
    baseline_entries: list[dict[str, object]] = []
    baseline_section = f"{FAMILY_SECTION}/{BASELINE_DELTA_SUBDIR}"
    baseline_dir = output_root / baseline_section
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for path in baseline_dir.glob("*"):
        if path.is_file() and path.suffix in {".pdf", ".png"}:
            path.unlink()
    for experiment in experiments:
        model_source = alternatives.factor_per_model_source(
            metrics, experiment, spider=spider
        )
        source = average_over_families(
            traces.trace_source(
                metrics, experiment, spider=spider, alternatives=alternatives
            ),
            traces=traces,
            spider=spider,
            model_source=model_source,
        )
        plot_family_traces(
            experiment=experiment,
            source=source,
            output_root=output_root,
            entries=entries,
            input_hashes=input_hashes,
            traces=traces,
            spider=spider,
        )
        plot_family_deltas(
            experiment=experiment,
            source=source,
            summary=traces.delta_summary(source),
            output_root=output_root,
            entries=entries,
            input_hashes=input_hashes,
            traces=traces,
            spider=spider,
        )
        delta_source = to_baseline_delta(
            source, model_source=model_source, traces=traces
        )
        plot_family_traces(
            experiment=experiment,
            source=delta_source,
            output_root=output_root,
            entries=baseline_entries,
            input_hashes=input_hashes,
            traces=traces,
            spider=spider,
            section=baseline_section,
            stem=f"{experiment}_outcome_family_baseline_delta_traces",
            description=(
                f"{spider.EXPERIMENT_DISPLAY[experiment]} 24-48h family-mean traces "
                "as Δ from each condition's baseline factor level; equal-weight mean "
                "of outcomes inside each clinical family after the 3-seed × "
                "2-architecture average."
            ),
            ylabel_lookup=traces.TASK_BASELINE_DELTA_LABELS,
            reference_y=0.0,
        )
    write_section_manifest(
        output_root=output_root,
        entries=entries,
        input_hashes=input_hashes,
    )
    metadata_md = write_baseline_delta_metadata(
        output_root=output_root,
        experiments=tuple(experiments),
        spider=spider,
    )
    write_section_manifest(
        output_root=output_root,
        entries=baseline_entries,
        input_hashes=input_hashes,
        section=baseline_section,
        extra={
            "parent_section": FAMILY_SECTION,
            "figure_metadata_markdown": str(metadata_md.relative_to(output_root)),
            "contract": {
                "y_axis": "delta_from_condition_baseline_family_mean",
                "aggregation": "equal_weight_mean_of_outcomes_within_clinical_family",
            },
        },
    )
    result = {
        "output_root": str(output_root),
        "section": FAMILY_SECTION,
        "baseline_delta_section": baseline_section,
        "experiments": list(experiments),
        "figure_count": len(entries),
        "baseline_delta_figure_count": len(baseline_entries),
    }
    if submission_figures_dir is not None:
        copied = copy_submission_traces(
            output_root=output_root,
            entries=entries,
            destination=submission_figures_dir,
            experiments=experiments,
        )
        result["submission_figures_dir"] = str(
            submission_figures_dir.expanduser().resolve()
        )
        result["submission_traces"] = copied
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metrics-long", type=Path, default=DEFAULT_METRICS_LONG)
    parser.add_argument(
        "--submission-figures-dir",
        type=Path,
        default=None,
        help="Copy exp1-trace-plot / exp2-trace-plot / exp3-trace-plot PDF/PNG/CSV into this paper figures folder.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=list(EXPERIMENTS),
    )
    args = parser.parse_args()
    result = generate(
        output_root=args.output_root.expanduser().resolve(),
        experiments=tuple(args.experiments),
        submission_figures_dir=(
            args.submission_figures_dir.expanduser().resolve()
            if args.submission_figures_dir is not None
            else None
        ),
        metrics_long=args.metrics_long.expanduser().resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

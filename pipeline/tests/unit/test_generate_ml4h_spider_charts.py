from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "paper"
    / "scripts"
    / "generate_ml4h_spider_charts.py"
)


def load_renderer():
    spec = importlib.util.spec_from_file_location(
        "generate_ml4h_spider_charts", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_three_figure_stems_palette_and_baselines() -> None:
    renderer = load_renderer()

    assert renderer.expected_figure_stems() == (
        "exp1_factor_spiders",
        "exp2_factor_spiders",
        "exp3_factor_spiders",
    )
    assert renderer.expected_table_stems() == (
        "exp1_factor_summary_table",
        "exp2_factor_summary_table",
        "exp3_factor_summary_table",
    )
    assert renderer.SERIES_COLORS[0] == "#E69F00"
    assert renderer.GRID_COLOR == "#5A5A5A"
    assert renderer.RING_COLOR == "#2A2A2A"
    assert renderer.PRIMARY_HORIZON == "24to48h"
    assert renderer.baseline_filters("exp1") == {
        "quantizer": "deciles",
        "anchoring": "none",
        "fusion": "unfused",
    }
    assert renderer.baseline_filters("exp2") == {
        "representation": "discrete",
        "temporal": "time_tokens",
    }
    assert renderer.baseline_filters("exp3") == {"source": "Native MIMIC"}


def test_panel_columns_split_conditions_not_horizons() -> None:
    renderer = load_renderer()

    exp1 = renderer.panel_columns("exp1")
    assert [column.key for column in exp1] == ["granularity", "fusion"]
    assert len(exp1[0].series) == 6
    assert len(exp1[1].series) == 2
    assert exp1[0].delta_grid.ticks == renderer.GRANULARITY_DELTA_TICKS
    assert exp1[1].delta_grid.ticks == renderer.FUSION_DELTA_TICKS
    assert renderer.FUSION_DELTA_TICKS == (-0.05, 0.00, 0.05, 0.10, 0.15)
    assert sum(
        item.filters == renderer.baseline_filters("exp1")
        for column in exp1
        for item in column.series
    ) == 2  # Deciles · population and Unfused are the same baseline, once per column

    exp2 = renderer.panel_columns("exp2")
    assert [column.key for column in exp2] == ["value_encoding", "temporal"]
    assert len(exp2[0].series) == 4
    assert [item.label for item in exp2[0].series] == [
        "Discrete",
        "Soft discretization",
        "xVal",
        "xVal-affine",
    ]
    assert any("xval_affine" in item.level_id for item in exp2[0].series)
    assert len(exp2[1].series) == 3
    assert [item.label for item in exp2[1].series] == [
        "Time tokens",
        "Event stream only",
        "Admission-relative RoPE",
    ]
    assert all(item.average_over for item in exp2[1].series)
    assert exp2[0].delta_grid.ticks == renderer.VALUE_ENCODING_DELTA_TICKS
    assert exp2[1].delta_grid.ticks == renderer.TEMPORAL_DELTA_TICKS
    assert sum(
        item.filters == renderer.baseline_filters("exp2")
        for column in exp2
        for item in column.series
    ) == 1  # Discrete only; Time tokens is an encoding-averaged baseline

    exp3 = renderer.panel_columns("exp3")
    assert [column.key for column in exp3] == ["source"]
    assert exp3[0].title == "Data schema"
    assert len(exp3[0].series) == 2


def test_unique_series_counts_and_baseline_membership() -> None:
    renderer = load_renderer()

    expected_counts = {"exp1": 7, "exp2": 7, "exp3": 2}
    for experiment, expected_count in expected_counts.items():
        series = renderer.experiment_series(experiment)
        assert len(series) == expected_count
        assert len({item.level_id for item in series}) == expected_count
        assert sum(
            item.filters == renderer.baseline_filters(experiment) for item in series
        ) == 1
        assert all(item.marker for item in series)
        assert all(item.linestyle for item in series)
    assert "xval_affine" in {
        item.filters.get("representation")
        for item in renderer.experiment_series("exp2")
    }


def test_exp2_temporal_averages_include_xval_affine() -> None:
    renderer = load_renderer()
    outcomes = ("same_admission_death",)
    rows = []
    for representation, base, event_order_bump, time_rope_bump in (
        ("discrete", 0.70, -0.04, 0.06),
        ("soft", 0.72, -0.04, 0.06),
        ("xval", 0.74, -0.04, 0.06),
        ("xval_affine", 0.90, -0.08, 0.09),
    ):
        for temporal, bump in (
            ("time_tokens", 0.00),
            ("event_order", event_order_bump),
            ("time_rope", time_rope_bump),
        ):
            rows.append(
                {
                    "config_id": f"{representation}|{temporal}",
                    "representation": representation,
                    "temporal": temporal,
                    "outcome": "same_admission_death",
                    "point": base + bump,
                }
            )
    panel = renderer.full_cross_delta_panel(
        pd.DataFrame(rows),
        experiment="exp2",
        series=renderer.exp2_temporal_series(),
        outcomes=outcomes,
    )
    by_label = panel.set_index("level_label")["delta"]
    assert np.isclose(by_label.loc["Time tokens"], 0.0)
    # Mean across Discrete, Soft, xVal, and xVal-affine.
    assert np.isclose(by_label.loc["Event stream only"], -0.05)
    assert np.isclose(by_label.loc["Admission-relative RoPE"], 0.0675)


def test_delta_radius_has_fixed_signed_grid_and_clipping() -> None:
    renderer = load_renderer()

    assert renderer.DELTA_TICKS == (-0.10, -0.05, 0.00, 0.05, 0.10)
    assert renderer.DATA_RADIUS_MAX == 1.12
    assert np.allclose(
        renderer.delta_tick_radii(),
        (0.0, 0.28, 0.56, 0.84, 1.12),
    )
    assert renderer.delta_to_radius(-0.10) == 0.0
    assert np.isclose(renderer.delta_to_radius(0.00), 0.56)
    assert renderer.delta_to_radius(0.10) == 1.12
    assert renderer.delta_to_radius(-0.32) == 0.0
    assert renderer.delta_to_radius(0.28) == 1.12
    assert renderer.format_delta(-0.125, precision=3) == "−0.125"
    assert renderer.format_delta(0.125, precision=3) == "+0.125"

    tight = renderer.DeltaGrid(renderer.GRANULARITY_DELTA_TICKS)
    assert tight.to_radius(0.0) == renderer.DATA_RADIUS_MAX / 2.0
    assert tight.to_radius(0.05) == renderer.DATA_RADIUS_MAX
    assert tight.is_outlier(0.06)
    assert not tight.is_outlier(0.05)

    asymmetric = renderer.DeltaGrid(renderer.VALUE_ENCODING_DELTA_TICKS)
    assert asymmetric.min == -0.15
    assert asymmetric.max == 0.05
    assert abs(float(asymmetric.to_radius(0.0)) - 0.84) < 1e-9
    assert asymmetric.is_outlier(0.06)
    assert asymmetric.is_outlier(-0.16)

    fusion = renderer.DeltaGrid(renderer.FUSION_DELTA_TICKS)
    assert fusion.ticks == (-0.05, 0.00, 0.05, 0.10, 0.15)
    assert abs(float(fusion.to_radius(0.0)) - 0.28) < 1e-9
    assert float(fusion.to_radius(0.15)) == renderer.DATA_RADIUS_MAX
    assert fusion.is_outlier(-0.06)
    assert fusion.is_outlier(0.16)


def test_outcome_label_radii_are_collision_aware() -> None:
    renderer = load_renderer()
    angles = np.linspace(-0.4, 0.4, 6)
    labels = (
        "Hospital\nmortality",
        "Long length\nof stay",
        "ICU\nadmission",
        "ICU stay\n>48h",
        "Invasive mechanical\nventilation",
        "Hyperkalemia",
    )
    radii = renderer.assign_nonoverlapping_label_radii(angles, labels)
    assert len(radii) == len(labels)
    assert radii[0] >= renderer.LABEL_RADIUS_BASE
    # Nearby labels use even/odd radial parity so adjacent spokes differ.
    assert len(set(round(radius, 3) for radius in radii)) >= 2
    assert renderer.DELTA_CAPTION_GAP >= 0.12
    assert renderer.series_is_baseline(
        "exp1", renderer.exp1_granularity_series()[0]
    )
    assert renderer.series_is_baseline("exp1", renderer.exp1_fusion_series()[0])
    assert renderer.series_is_baseline(
        "exp2", renderer.exp2_value_encoding_series()[0]
    )
    assert renderer.series_is_baseline("exp2", renderer.exp2_temporal_series()[0])
    value_handles = renderer._legend_handles(
        renderer.exp2_value_encoding_series(), experiment="exp2"
    )
    assert value_handles[0].get_label() == "Discrete (baseline)"
    assert value_handles[1].get_label() == "Soft discretization"
    temporal_handles = renderer._legend_handles(
        renderer.exp2_temporal_series(), experiment="exp2"
    )
    assert temporal_handles[0].get_label() == "Time tokens (baseline)"
    assert temporal_handles[1].get_label() == "Event stream only"


def test_binary_and_regression_spokes_use_opposite_halves() -> None:
    renderer = load_renderer()

    binary_angles, regression_angles, all_angles = renderer.half_circle_angles(
        ("b1", "b2", "b3"),
        ("r1", "r2", "r3", "r4"),
    )
    # With theta zero at North, visual y is proportional to cos(theta).
    assert np.all(np.cos(binary_angles) > 0.0)
    assert np.all(np.cos(regression_angles) < 0.0)
    assert len(all_angles) == 7


def test_seed_mean_deltas_are_outcome_specific_to_baseline() -> None:
    renderer = load_renderer()
    rows = pd.DataFrame(
        [
            {
                "config_id": "native",
                "source": "Native MIMIC",
                "outcome": outcome,
                "point": point,
            }
            for outcome, points in {
                "same_admission_death": (0.70, 0.72),
                "length_of_stay": (0.30, 0.32),
            }.items()
            for point in points
        ]
        + [
            {
                "config_id": "clif",
                "source": "CLIF harmonized",
                "outcome": outcome,
                "point": point,
            }
            for outcome, points in {
                "same_admission_death": (0.77, 0.79),
                "length_of_stay": (0.21, 0.23),
            }.items()
            for point in points
        ]
    )

    panel = renderer.full_cross_delta_panel(
        rows,
        experiment="exp3",
        series=renderer.experiment_series("exp3"),
        outcomes=("same_admission_death", "length_of_stay"),
    )
    native = panel.loc[panel.level_id.eq("Native MIMIC")]
    clif = panel.loc[panel.level_id.eq("CLIF harmonized")].set_index("outcome")
    assert np.allclose(native.delta, 0.0)
    assert np.isclose(clif.loc["same_admission_death", "delta"], 0.07)
    assert np.isclose(clif.loc["length_of_stay", "delta"], -0.09)
    assert not bool(clif.loc["length_of_stay", "is_outlier"])


def test_summary_table_uses_condition_rows_and_24to48h_only() -> None:
    renderer = load_renderer()
    rows = []
    for column in renderer.panel_columns("exp3"):
        for item in column.series:
            for backbone in renderer.BACKBONE_ORDER:
                for task_type in renderer.TASK_ORDER:
                    rows.append(
                        {
                            "level_id": item.level_id,
                            "condition": column.key,
                            "backbone": backbone,
                            "horizon": renderer.PRIMARY_HORIZON,
                            "task_type": task_type,
                            "delta": 0.0
                            if item.filters
                            == renderer.baseline_filters("exp3")
                            else 0.025,
                        }
                    )
    summary = renderer.summary_mean_table(pd.DataFrame(rows), "exp3")
    assert len(summary) == 2
    assert set(summary.configuration) == {"Native MIMIC", "CLIF harmonized"}
    expected_columns = {
        renderer._summary_column(backbone, task_type)
        for backbone in renderer.BACKBONE_ORDER
        for task_type in renderer.TASK_ORDER
    }
    assert expected_columns.issubset(summary.columns)


def test_summary_table_text_fits_cells_without_overlapping_title_band() -> None:
    renderer = load_renderer()
    rows = []
    for column in renderer.panel_columns("exp1"):
        for item in column.series:
            for backbone in renderer.BACKBONE_ORDER:
                for task_type in renderer.TASK_ORDER:
                    rows.append(
                        {
                            "level_id": item.level_id,
                            "condition": column.key,
                            "backbone": backbone,
                            "task_type": task_type,
                            "delta": 0.0,
                        }
                    )
    figure = renderer.build_summary_table_figure(pd.DataFrame(rows), experiment="exp1")
    try:
        figure.canvas.draw()
        canvas_renderer = figure.canvas.get_renderer()
        title_bounds = figure._summary_title.get_window_extent(canvas_renderer)
        subtitle_bounds = figure._summary_subtitle.get_window_extent(canvas_renderer)
        assert title_bounds.y0 > subtitle_bounds.y1
        for (row_index, column_index), cell in figure._summary_table.get_celld().items():
            text_bounds = cell.get_text().get_window_extent(canvas_renderer)
            cell_bounds = cell.get_window_extent(canvas_renderer)
            assert text_bounds.height <= cell_bounds.height * 0.93, (
                row_index,
                column_index,
            )
            assert text_bounds.width <= cell_bounds.width * 0.96, (
                row_index,
                column_index,
            )
    finally:
        renderer.plt.close(figure)


def test_experiment_figure_panel_counts_by_experiment() -> None:
    renderer = load_renderer()

    def _rows(experiment: str) -> pd.DataFrame:
        rows = []
        for backbone in renderer.BACKBONE_ORDER:
            for task_type, outcomes in (
                ("binary", renderer.BINARY_ORDER),
                ("regression", renderer.REGRESSION_ORDER),
            ):
                metric = renderer.METRIC_BY_TASK[task_type]
                for column in renderer.panel_columns(experiment):
                    for item in column.series:
                        for outcome in outcomes:
                            row = {
                                "experiment": experiment,
                                "backbone": backbone,
                                "horizon": renderer.PRIMARY_HORIZON,
                                "task_type": task_type,
                                "metric": metric,
                                "config_id": item.level_id,
                                "outcome": outcome,
                                "outcome_label": renderer.FULL_OUTCOME_LABELS[outcome],
                                "point": 0.70,
                            }
                            row.update(item.filters)
                            rows.append(row)
        return pd.DataFrame(rows)

    renderer.style()
    for experiment, expected_panels in (("exp1", 4), ("exp2", 4), ("exp3", 2)):
        figure, source = renderer.build_experiment_figure(
            _rows(experiment), experiment=experiment
        )
        try:
            polar = [axis for axis in figure.axes if axis.name == "polar"]
            assert len(polar) == expected_panels
            assert set(source.task_type) == {"binary", "regression"}
            assert set(source.horizon) == {renderer.PRIMARY_HORIZON}
            assert "restofstay" not in set(source.horizon)
            assert set(source.condition) == {
                column.key for column in renderer.panel_columns(experiment)
            }
            if experiment == "exp3":
                # Side-by-side Llama | Qwen rather than stacked.
                assert figure.get_figwidth() > figure.get_figheight()
        finally:
            renderer.plt.close(figure)

    assert renderer.PANEL_LETTERS == ("A", "B", "C", "D")
    assert "Hospital mortality in hours 24--48" in renderer.FULL_OUTCOME_LABELS.values()
    wrapped = renderer.wrap_outcome_label(
        "Continuous renal replacement therapy", width=18
    )
    assert "\n" in wrapped
    assert renderer.NEURIPS_BODY_PT == 24.0
    assert renderer.PANEL_LETTER_PT == 40.0
    assert renderer.LEGEND_PT >= 20.0
    assert renderer.OUTCOME_LABEL_FONTSIZE >= 18.0
    assert renderer.OUTCOME_KEY_PT >= 18.0
    assert renderer.FIGSIZE_EXP12[0] <= 16.0
    assert all(
        label.endswith("-style") for label in renderer.BACKBONE_DISPLAY.values()
    )
    assert renderer.FULL_OUTCOME_LABELS["peak_creatinine"] == "Maximum creatinine"
    assert renderer.FULL_OUTCOME_LABELS["crrt_initiation"] == (
        "Continuous renal replacement"
    )
    assert "HyperK" not in renderer.FULL_OUTCOME_LABELS.values()
    assert "Min Cr" not in renderer.FULL_OUTCOME_LABELS.values()
    binary_lines, regression_lines = renderer.outcome_key_entries(
        ("same_admission_death", "icu_admission"),
        ("min_potassium", "min_creatinine"),
        {},
    )
    assert binary_lines[0].replace("\n    ", " ").startswith(
        "1  Hospital mortality in hours 24--48"
    )
    assert regression_lines[0].startswith("3  Minimum potassium")
    assert "Minimum creatinine" in regression_lines[1]
    assert renderer.OUTCOME_FAMILIES["crrt_initiation"] == "Organ support"
    assert renderer.OUTCOME_FAMILIES["min_glucose"] == "Biomarkers"
    assert renderer.BINARY_ORDER.index("imv_event") < renderer.BINARY_ORDER.index(
        "hyperkalemia"
    )
    assert renderer.BINARY_ORDER.index("vasopressor_initiation") < renderer.BINARY_ORDER.index(
        "hyperkalemia"
    )
    assert "\n" in renderer.outcome_key_entries(
        ("crrt_initiation",),
        (),
        {},
        wrap_width=18,
    )[0][0]
    assert any(
        column.title == "Numeric representation"
        for column in renderer.panel_columns("exp2")
    )
    assert "Value encoding" not in {
        column.title for column in renderer.panel_columns("exp2")
    }


def test_figure_metadata_markdown_holds_titles_and_full_legends(tmp_path) -> None:
    renderer = load_renderer()
    path = renderer.write_figure_metadata_markdown(
        section_dir=tmp_path,
        experiments=("exp1", "exp2", "exp3"),
    )
    text = path.read_text(encoding="utf-8")
    assert "Spider chart figure metadata" in text
    assert "Experiment 1: numeric tokenization" in text
    assert "Deciles · population" in text
    assert "Soft discretization" in text
    assert "Discrete (baseline)" in text
    assert "Unfused (baseline)" in text
    assert "Time tokens (baseline)" in text
    assert "Native MIMIC (baseline)" in text
    assert "Admission-relative RoPE" in text
    assert "Event stream only" in text
    assert "Data schema" in text
    assert "Time-aware RoPE" not in text
    assert "Event order" not in text
    assert "Data source" not in text
    assert "Δ vs baseline" not in text.replace("relative to baseline", "")
    assert "Numeric representation" in text
    assert "Value encoding" not in text
    assert "D-pop" not in text
    assert "Disc" not in text.split("Discrete")[0]  # no bare Disc abbreviation left as label
    assert "24–48h follow-up" in text

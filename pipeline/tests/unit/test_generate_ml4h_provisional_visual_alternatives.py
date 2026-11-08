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
    / "generate_ml4h_provisional_visual_alternatives.py"
)


def load_renderer():
    spec = importlib.util.spec_from_file_location(
        "generate_ml4h_provisional_visual_alternatives", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def exp2_metrics(renderer) -> pd.DataFrame:
    primary = renderer.load_primary_renderer()
    rows = []
    for config_index, config_id in enumerate(primary.EXP2_CONFIG_ORDER):
        fields = primary.experiment_config_parts(
            primary.EXPERIMENT_SPECS["exp2"], config_id
        )
        for backbone in ("llama", "qwen"):
            for seed in (42, 43, 44):
                for task_type, outcome, metric in (
                    ("binary", "same_admission_death", "roc_auc"),
                    ("regression", "min_creatinine", "spearman_rho"),
                ):
                    rows.append(
                        {
                            "experiment": "exp2",
                            "config_id": config_id,
                            "backbone": backbone,
                            "seed": seed,
                            "task_type": task_type,
                            "outcome": outcome,
                            "metric": metric,
                            "point": 0.60 + config_index * 0.001 + seed * 0.00001,
                            **fields,
                        }
                    )
    return pd.DataFrame(rows)


def test_exp2_absolute_display_includes_affine_and_orders_time_tokens_first() -> None:
    renderer = load_renderer()
    primary = renderer.load_primary_renderer()
    spider = renderer.load_spider_renderer()

    source, config_order = renderer.seed_interval_source(
        exp2_metrics(renderer),
        "exp2",
        primary=primary,
        spider=spider,
    )

    assert len(config_order) == 12
    assert any("xval_affine" in config for config in config_order)
    labels = (
        source[["config_id", "config_label"]]
        .drop_duplicates()
        .set_index("config_id")["config_label"]
    )
    assert labels.iloc[0] == "Discrete · Time tokens (baseline)"
    assert labels.iloc[1] == "Discrete · Event stream only"
    assert labels.iloc[2] == "Discrete · Admission-relative RoPE"
    assert any(label.startswith("Soft discretization ·") for label in labels)
    assert source["outcome_label"].eq("Hospital mortality in hours 24--48").any()


def test_factor_source_uses_spider_temporal_average_with_affine() -> None:
    renderer = load_renderer()
    spider = renderer.load_spider_renderer()

    source = renderer.factor_seed_interval_source(
        exp2_metrics(renderer),
        "exp2",
        spider=spider,
    )

    temporal = source.loc[source["condition"] == "temporal"]
    assert set(temporal["level_label"]) == {
        "Time tokens",
        "Event stream only",
        "Admission-relative RoPE",
    }
    assert temporal["component_config_ids"].str.contains("xval_affine").all()
    assert temporal["n_seeds"].eq(3).all()
    assert temporal["outcome_label"].eq("Minimum creatinine").any()


def test_aggregate_delta_source_keeps_baseline_win_rate_undefined() -> None:
    renderer = load_renderer()
    spider = renderer.load_spider_renderer()
    source = pd.DataFrame(
        [
            {
                "experiment": "exp3",
                "condition": "source",
                "level_id": "Native MIMIC",
                "level_label": "Native MIMIC",
                "is_baseline": True,
                "backbone": "llama",
                "task_type": "binary",
                "metric": "roc_auc",
                "outcome": "same_admission_death",
                "delta": 0.0,
            },
            {
                "experiment": "exp3",
                "condition": "source",
                "level_id": "CLIF harmonized",
                "level_label": "CLIF harmonized",
                "is_baseline": False,
                "backbone": "llama",
                "task_type": "binary",
                "metric": "roc_auc",
                "outcome": "same_admission_death",
                "delta": 0.02,
            },
            {
                "experiment": "exp3",
                "condition": "source",
                "level_id": "CLIF harmonized",
                "level_label": "CLIF harmonized",
                "is_baseline": False,
                "backbone": "llama",
                "task_type": "binary",
                "metric": "roc_auc",
                "outcome": "long_length_of_stay",
                "delta": -0.01,
            },
        ]
    )

    aggregate = renderer.aggregate_delta_source(source, spider=spider).set_index(
        "level_id"
    )
    assert np.isnan(aggregate.loc["Native MIMIC", "win_rate"])
    assert np.isclose(aggregate.loc["CLIF harmonized", "mean_delta"], 0.005)
    assert np.isclose(aggregate.loc["CLIF harmonized", "win_rate"], 0.5)


def test_seed_interval_forest_panel_matches_spider_legibility() -> None:
    renderer = load_renderer()
    spider = renderer.load_spider_renderer()
    source, config_order = renderer.seed_interval_source(
        exp2_metrics(renderer),
        "exp2",
        primary=renderer.load_primary_renderer(),
        spider=spider,
    )
    figure, axis = renderer.plt.subplots(figsize=(8, 10))
    try:
        subset = source.loc[source["backbone"] == "llama"]
        outcomes = renderer.outcome_order_for_source(
            subset, "binary", spider=spider
        )
        renderer.draw_config_forest_panel(
            axis,
            subset.rename(columns={"config_id": "level_id"}),
            level_ids=config_order,
            outcomes=outcomes,
            task_type="binary",
            title="Binary AUROC",
            panel_letter="A",
            match_spider_legibility=True,
            spider=spider,
        )
        letters = [
            text.get_text()
            for text in axis.texts
            if text.get_text() in spider.PANEL_LETTERS
        ]
        assert letters == ["A"]
        assert axis.title.get_fontsize() == spider.COLUMN_TITLE_PT
        assert axis.get_xlabel() == ""
        ytick_sizes = {label.get_fontsize() for label in axis.get_yticklabels()}
        assert ytick_sizes == {spider.OUTCOME_KEY_PT}
        letter_artist = next(
            text for text in axis.texts if text.get_text() == "A"
        )
        assert letter_artist.get_position()[1] > 1.0
        assert renderer.forest_outcome_label(
            "crrt_initiation", spider=spider
        ) == "Continuous renal replacement"
        assert renderer.forest_outcome_label("min_bnp", spider=spider) == (
            "Min. natriuretic peptide"
        )
        assert renderer.forest_outcome_label("peak_bnp", spider=spider) == (
            "Max. natriuretic peptide"
        )
        assert renderer.forest_outcome_label(
            "min_creatinine", spider=spider
        ) == "Min. creatinine"
    finally:
        renderer.plt.close(figure)


def test_factor_bar_panel_shapes_and_outcome_contracts() -> None:
    renderer = load_renderer()
    spider = renderer.load_spider_renderer()

    assert renderer.factor_bar_panel_shape("exp1", spider=spider) == (2, 2)
    assert renderer.factor_bar_panel_shape("exp2", spider=spider) == (2, 2)
    assert renderer.factor_bar_panel_shape("exp3", spider=spider) == (1, 2)

    provisional_regression = tuple(
        outcome
        for outcome in spider.REGRESSION_ORDER
        if outcome not in {"length_of_stay", "icu_length_of_stay"}
    )
    full = pd.DataFrame(
        [
            {
                "task_type": task_type,
                "outcome": outcome,
            }
            for task_type, outcomes in (
                ("binary", spider.BINARY_ORDER),
                ("regression", provisional_regression),
            )
            for outcome in outcomes
        ]
    )
    exp3 = full.loc[full["outcome"] != "icu_admission"]
    assert len(renderer.factor_bar_outcomes(full, spider=spider)[2]) == 37
    assert len(renderer.factor_bar_outcomes(exp3, spider=spider)[2]) == 36


def test_factor_bar_panel_uses_spider_membership_divider_and_45_degree_labels() -> None:
    renderer = load_renderer()
    spider = renderer.load_spider_renderer()
    source = renderer.factor_seed_interval_source(
        exp2_metrics(renderer),
        "exp2",
        spider=spider,
    )
    column = spider.panel_columns("exp2")[0]
    figure, axis = renderer.plt.subplots(figsize=(10, 6))
    try:
        renderer.draw_factor_bar_panel(
            axis,
            source,
            experiment="exp2",
            column=column,
            backbone="llama",
            panel_letter="A",
            show_xlabels=True,
            spider=spider,
        )
        outcomes = axis._factor_bar_outcomes
        assert len(axis.patches) == len(outcomes) * len(column.series)
        centers = [
            axis.patches[series_index * len(outcomes)].get_x()
            + axis.patches[series_index * len(outcomes)].get_width() / 2.0
            for series_index in range(len(column.series))
        ]
        assert np.allclose(centers, 0.0)
        assert axis._factor_bar_divider.get_linewidth() >= 2.6
        assert all(label.get_rotation() == 45 for label in axis.get_xticklabels())
        assert source["component_config_ids"].str.contains("xval_affine").any()
    finally:
        renderer.plt.close(figure)


def test_factor_trace_extremes_use_seed_means_within_panel() -> None:
    renderer = load_renderer()
    spider = renderer.load_spider_renderer()
    source = renderer.factor_seed_interval_source(
        exp2_metrics(renderer),
        "exp2",
        spider=spider,
    )
    column = spider.panel_columns("exp2")[0]
    subset = source.loc[
        (source["backbone"] == "llama")
        & (source["condition"] == column.key)
        & (source["task_type"] == "binary")
    ]
    outcomes = renderer.outcome_order_for_source(subset, "binary", spider=spider)
    extremes = renderer.factor_trace_extremes(
        subset,
        outcomes=outcomes,
        level_ids=tuple(item.level_id for item in column.series),
    )
    assert renderer.factor_trace_panel_shape("exp1", spider=spider) == (2, 2)
    assert renderer.factor_trace_panel_shape("exp3", spider=spider) == (1, 2)
    assert set(extremes["outcome"]) == set(outcomes)
    assert (extremes["spread"] >= 0).all()
    assert extremes["best_level_id"].ne(extremes["worst_level_id"]).all()
    assert source["component_config_ids"].str.contains("xval_affine").any()
    figure, axis = renderer.plt.subplots(figsize=(8, 5))
    try:
        renderer.draw_factor_trace_summary(
            axis,
            outcomes=outcomes,
            extremes=extremes,
            show_xlabels=True,
            spider=spider,
        )
        labels = [text.get_text() for text in axis.get_yticklabels()]
        assert all(" " not in label or label == label for label in labels)
        assert all("\n" not in label for label in labels)
        assert any("Hospital mortality in hours 24--48" == label for label in labels)
    finally:
        renderer.plt.close(figure)


def test_factor_forests_average_over_architectures() -> None:
    renderer = load_renderer()
    spider = renderer.load_spider_renderer()
    rows = []
    for backbone, point, lo, hi in (
        ("llama", 0.80, 0.78, 0.82),
        ("qwen", 0.70, 0.68, 0.73),
    ):
        rows.append(
            {
                "experiment": "exp2",
                "backbone": backbone,
                "condition": "value_encoding",
                "condition_title": "Numeric representation",
                "task_type": "binary",
                "metric": "roc_auc",
                "outcome": "same_admission_death",
                "outcome_label": "Hospital mortality in hours 24--48",
                "level_id": "discrete|time_tokens",
                "level_label": "Discrete",
                "color": "#0072B2",
                "marker": "o",
                "linestyle": "solid",
                "is_baseline": True,
                "component_config_ids": "meds_deciles_none_fusedFalse_discrete_time_tokens",
                "point": point,
                "lo": lo,
                "hi": hi,
                "n_seeds": 3,
            }
        )
    source = pd.DataFrame(rows)
    pooled = renderer.average_factor_source_over_architectures(
        source, spider=spider
    )
    assert list(pooled["backbone"]) == [renderer.POOLED_BACKBONE]
    assert len(pooled) == 1
    assert int(pooled.loc[0, "n_architectures"]) == 2
    assert pooled.loc[0, "point"] == 0.75
    assert pooled.loc[0, "lo"] == 0.68
    assert pooled.loc[0, "hi"] == 0.82

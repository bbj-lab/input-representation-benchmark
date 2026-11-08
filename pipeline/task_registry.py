"""Single source of truth for ML4H clinical probe task families."""

from __future__ import annotations


BINARY_TASKS = (
    "same_admission_death",
    "long_length_of_stay",
    "icu_admission",
    "imv_event",
    "prolonged_icu_stay",
    "hyperkalemia",
    "severe_hypokalemia",
    "severe_anemia",
    "hypoglycemia",
    "profound_hyponatremia",
    "severe_hypernatremia",
    "tachycardia_hr130",
    "severe_hypertension",
    "vasopressor_initiation",
    "hypotension",
    "crrt_initiation",
    "hemodialysis_initiation",
)

REGRESSION_TASKS = (
    "length_of_stay",
    "icu_length_of_stay",
    "min_creatinine",
    "peak_creatinine",
    "min_troponin",
    "peak_troponin",
    "min_hemoglobin",
    "max_hemoglobin",
    "min_potassium",
    "peak_potassium",
    "min_glucose",
    "max_glucose",
    "min_sodium",
    "max_sodium",
    "min_bnp",
    "peak_bnp",
    "min_heart_rate",
    "max_heart_rate",
    "min_sbp",
    "max_sbp",
    "min_dbp",
    "max_dbp",
)

TASK_GROUPS = {
    "binary": BINARY_TASKS,
    "regression": REGRESSION_TASKS,
}


def task_names(task_type: str) -> tuple[str, ...]:
    try:
        return TASK_GROUPS[task_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown task type {task_type!r}; expected one of {tuple(TASK_GROUPS)}."
        ) from exc

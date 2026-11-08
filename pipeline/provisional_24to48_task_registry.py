"""Task contract for the isolated ML4H 24-to-48-hour provisional analysis."""

from __future__ import annotations

from pipeline.task_registry import BINARY_TASKS, REGRESSION_TASKS


HORIZON_NAME = "24to48h"
HORIZON_START_HOURS = 24.0
HORIZON_END_HOURS = 48.0
OUTCOMES_FILENAME = "tokens_timelines_extended_outcomes_24to48h.parquet"

# The two discharge-duration regression labels cannot be confined to a 24-hour
# follow-up interval, so they are deliberately not probed in this analysis.
OMITTED_REGRESSION_TASKS = ("length_of_stay", "icu_length_of_stay")
PROVISIONAL_REGRESSION_TASKS = tuple(
    task for task in REGRESSION_TASKS if task not in OMITTED_REGRESSION_TASKS
)
PROVISIONAL_BINARY_TASKS = BINARY_TASKS

# Exp3 uses the ICU-restricted cohort, where ICU admission is degenerate.
PROVISIONAL_EXP3_BINARY_TASKS = tuple(
    task for task in PROVISIONAL_BINARY_TASKS if task != "icu_admission"
)

EXPECTED_TASK_COUNTS = {
    "exp1": len(PROVISIONAL_BINARY_TASKS) + len(PROVISIONAL_REGRESSION_TASKS),
    "exp2": len(PROVISIONAL_BINARY_TASKS) + len(PROVISIONAL_REGRESSION_TASKS),
    "exp3": len(PROVISIONAL_EXP3_BINARY_TASKS)
    + len(PROVISIONAL_REGRESSION_TASKS),
}

TASK_SEMANTICS = {
    "same_admission_death": "death during (24h, 48h] after admission",
    "long_length_of_stay": "still hospitalized at hour 48",
    "prolonged_icu_stay": "still in ICU at hour 48",
    "icu_admission": "first ICU admission during (24h, 48h]",
    "imv_event": "first IMV procedure during (24h, 48h]",
}

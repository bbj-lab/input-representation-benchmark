from __future__ import annotations

from pipeline.provisional_24to48_task_registry import (
    EXPECTED_TASK_COUNTS,
    OMITTED_REGRESSION_TASKS,
    PROVISIONAL_BINARY_TASKS,
    PROVISIONAL_EXP3_BINARY_TASKS,
    PROVISIONAL_REGRESSION_TASKS,
)


def test_provisional_task_contract_has_expected_counts() -> None:
    assert len(PROVISIONAL_BINARY_TASKS) == 17
    assert len(PROVISIONAL_REGRESSION_TASKS) == 20
    assert len(PROVISIONAL_EXP3_BINARY_TASKS) == 16
    assert EXPECTED_TASK_COUNTS == {"exp1": 37, "exp2": 37, "exp3": 36}


def test_provisional_contract_omits_only_unbounded_regression_tasks() -> None:
    assert OMITTED_REGRESSION_TASKS == ("length_of_stay", "icu_length_of_stay")
    assert not set(OMITTED_REGRESSION_TASKS) & set(PROVISIONAL_REGRESSION_TASKS)
    assert "icu_admission" not in PROVISIONAL_EXP3_BINARY_TASKS

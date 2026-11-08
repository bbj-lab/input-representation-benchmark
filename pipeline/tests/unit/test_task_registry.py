from pipeline.task_registry import BINARY_TASKS, REGRESSION_TASKS, task_names


def test_task_registry_partitions_tasks_by_prediction_type():
    assert task_names("binary") == BINARY_TASKS
    assert task_names("regression") == REGRESSION_TASKS
    assert set(BINARY_TASKS).isdisjoint(REGRESSION_TASKS)


def test_regression_task_registry_includes_symmetric_extrema():
    for minimum, maximum in (
        ("min_creatinine", "peak_creatinine"),
        ("min_troponin", "peak_troponin"),
        ("min_hemoglobin", "max_hemoglobin"),
        ("min_glucose", "max_glucose"),
        ("min_sodium", "max_sodium"),
        ("min_bnp", "peak_bnp"),
        ("min_heart_rate", "max_heart_rate"),
        ("min_sbp", "max_sbp"),
        ("min_dbp", "max_dbp"),
    ):
        assert minimum in REGRESSION_TASKS
        assert maximum in REGRESSION_TASKS

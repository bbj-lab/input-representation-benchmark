from datetime import datetime

import polars as pl

from pipeline.scripts.convert_clif_to_meds import convert_split


def test_convert_clif_to_meds_preserves_raw_cohort_assignment(tmp_path):
    source = tmp_path / "raw_meds"
    (source / "train").mkdir(parents=True)
    pl.DataFrame(
        {
            "subject_id": [10, 10, 20],
            "hadm_id": [100, 100, 200],
            "time": [datetime(2020, 1, 1)] * 3,
            "code": ["HOSPITAL_ADMISSION//x"] * 3,
            "numeric_value": [None, None, None],
        }
    ).write_parquet(source / "train" / "meds.parquet")

    clif = tmp_path / "clif"
    clif.mkdir()
    pl.DataFrame(
        {
            "patient_id": ["10", "20", "30"],
            "hospitalization_id": ["100", "200", "300"],
            "admission_dttm": [datetime(2020, 1, 1)] * 3,
            "discharge_dttm": [datetime(2020, 1, 3)] * 3,
            "admission_type_category": ["emergency"] * 3,
            "discharge_category": ["home"] * 3,
        }
    ).write_parquet(clif / "clif_hospitalization.parquet")
    pl.DataFrame(
        {
            "hospitalization_id": ["100", "300"],
            "lab_result_dttm": [datetime(2020, 1, 2)] * 2,
            "lab_category": ["chemistry", "chemistry"],
            "lab_name": ["potassium", "potassium"],
            "lab_value_numeric": [4.2, 5.1],
        }
    ).write_parquet(clif / "clif_labs.parquet")

    output = tmp_path / "clif_meds"
    summary = convert_split(
        clif_dir=clif,
        source_meds_dir=source,
        output_dir=output,
        split="train",
    )
    rows = pl.read_parquet(output / "train" / "meds.parquet")

    assert summary["cohort_admissions"] == 2
    assert set(rows["hadm_id"]) == {100, 200}
    assert "CLIF_LABS//chemistry//potassium" in set(rows["code"])

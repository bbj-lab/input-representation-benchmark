from pathlib import Path

import polars as pl

from pipeline.scripts.split_meds_by_hadm_splits import main


def _write_source_split(path: Path, subject_id: int, hadm_id: int) -> None:
    path.mkdir(parents=True)
    pl.DataFrame(
        {
            "subject_id": [subject_id],
            "hadm_id": [hadm_id],
            "time": [None],
            "code": ["HOSPITAL_ADMISSION//ed"],
        },
        schema={
            "subject_id": pl.Int64,
            "hadm_id": pl.Int64,
            "time": pl.Datetime,
            "code": pl.String,
        },
    ).write_parquet(path / "events.parquet")


def test_exp3_split_searches_all_source_splits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source_split(source / "train", 103, 1003)
    _write_source_split(source / "tuning", 101, 1001)
    _write_source_split(source / "test", 102, 1002)

    splits = tmp_path / "splits"
    splits.mkdir()
    for split, hadm_id in (("train", 1001), ("val", 1002), ("test", 1003)):
        pl.DataFrame({"hadm_id": [hadm_id]}).write_csv(
            splits / f"{split}_hadm_ids.csv"
        )

    output = tmp_path / "output"
    assert (
        main(
            [
                "--meds_in_dir",
                str(source),
                "--splits_dir",
                str(splits),
                "--meds_out_dir",
                str(output),
            ]
        )
        == 0
    )
    for split, hadm_id in (("train", 1001), ("val", 1002), ("test", 1003)):
        actual = pl.read_parquet(output / split / "meds.parquet")
        assert actual.get_column("hadm_id").to_list() == [hadm_id]

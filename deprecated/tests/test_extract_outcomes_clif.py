from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from scripts.extract_outcomes_clif import main as extract_main


class TestExtractOutcomesClif(unittest.TestCase):
    def test_extract_outcomes_clif_basic(self) -> None:
        with TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "raw" / "train").mkdir(parents=True)
            (tmp / "day_stays_first_24h-tokenized" / "train").mkdir(parents=True)

            t0 = datetime(2020, 1, 1, 0, 0, 0)

            # Raw hospitalization table
            pl.DataFrame(
                [
                    {
                        "hospitalization_id": 1,
                        "admission_dttm": t0,
                        "discharge_dttm": t0 + timedelta(hours=100),
                        "discharge_category": "home",
                    },
                    {
                        "hospitalization_id": 2,
                        "admission_dttm": t0,
                        "discharge_dttm": t0 + timedelta(hours=40),
                        "discharge_category": "expired",
                    },
                ]
            ).write_parquet(tmp / "raw" / "train" / "clif_hospitalization.parquet")

            # ICU ADT segments: hosp 1 has 60h ICU (prolonged); hosp 2 has 10h ICU
            pl.DataFrame(
                [
                    {
                        "hospitalization_id": 1,
                        "in_dttm": t0 + timedelta(hours=5),
                        "out_dttm": t0 + timedelta(hours=65),
                        "location_category": "ICU",
                    },
                    {
                        "hospitalization_id": 2,
                        "in_dttm": t0 + timedelta(hours=1),
                        "out_dttm": t0 + timedelta(hours=11),
                        "location_category": "icu",
                    },
                ]
            ).write_parquet(tmp / "raw" / "train" / "clif_adt.parquet")

            # Respiratory support: hosp 2 has IMV at hour 2
            pl.DataFrame(
                [
                    {
                        "hospitalization_id": 2,
                        "recorded_dttm": t0 + timedelta(hours=2),
                        "mode_category": "IMV",
                        "device_category": None,
                    }
                ]
            ).write_parquet(tmp / "raw" / "train" / "clif_respiratory_support_processed.parquet")

            # Tokenized timelines (only id is required for join)
            pl.DataFrame(
                [{"hospitalization_id": "1", "tokens": [1], "times": [t0]}, {"hospitalization_id": "2", "tokens": [2], "times": [t0]}]
            ).write_parquet(tmp / "day_stays_first_24h-tokenized" / "train" / "tokens_timelines.parquet")

            rc = extract_main(
                [
                    "--data_dir",
                    str(tmp),
                    "--ref_version",
                    "day_stays",
                    "--data_version",
                    "day_stays_first_24h",
                    "--splits",
                    "train",
                    "--icu_los_hours_threshold",
                    "48",
                ]
            )
            self.assertEqual(rc, 0)

            out = pl.read_parquet(tmp / "day_stays_first_24h-tokenized" / "train" / "tokens_timelines_outcomes.parquet").sort(
                "hospitalization_id"
            )
            self.assertEqual(out["hospitalization_id"].to_list(), ["1", "2"])
            row1 = out.filter(pl.col("hospitalization_id") == "1").row(0, named=True)
            self.assertFalse(row1["same_admission_death"])
            self.assertTrue(row1["icu_admission"])
            self.assertTrue(row1["prolonged_icu_stay"])
            row2 = out.filter(pl.col("hospitalization_id") == "2").row(0, named=True)
            self.assertTrue(row2["same_admission_death"])
            self.assertTrue(row2["imv_event"])
            self.assertTrue(row2["imv_event_24h"])


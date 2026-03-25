from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from pipeline.scripts.extract_outcomes_meds import main as extract_main


class TestExtractOutcomesMeds(unittest.TestCase):
    def test_extract_outcomes_meds_basic(self) -> None:
        with TemporaryDirectory() as d:
            tmp_path = Path(d)
            meds_events_dir = tmp_path / "meds_events"
            tokenized_dir = tmp_path / "tokenized"

            (meds_events_dir / "train").mkdir(parents=True)
            (tokenized_dir / "train").mkdir(parents=True)

            t0 = datetime(2020, 1, 1, 0, 0, 0)

            meds = pl.DataFrame(
                [
                    # hadm 100 (subject 1): ICU after 24h, IMV within 24h, survives
                    {
                        "subject_id": 1,
                        "hadm_id": 100,
                        "time": t0,
                        "code": "HOSPITAL_ADMISSION//EMERGENCY",
                    },
                    {
                        "subject_id": 1,
                        "hadm_id": 100,
                        "time": t0 + timedelta(hours=48),
                        "code": "HOSPITAL_DISCHARGE//HOME",
                    },
                    {
                        "subject_id": 1,
                        "hadm_id": 100,
                        "time": t0 + timedelta(hours=30),
                        "code": "ICU_ADMISSION//MICU",
                    },
                    {
                        "subject_id": 1,
                        "hadm_id": 100,
                        "time": t0 + timedelta(hours=10),
                        "code": "PROCEDURE//225792",
                    },
                    # Patient death occurs after discharge: should not set same_admission_death
                    {
                        "subject_id": 1,
                        "hadm_id": None,
                        "time": t0 + timedelta(days=10),
                        "code": "MEDS_DEATH",
                    },
                    # hadm 200 (subject 2): dies during admission, no ICU/IMV
                    {
                        "subject_id": 2,
                        "hadm_id": 200,
                        "time": t0,
                        "code": "HOSPITAL_ADMISSION//EMERGENCY",
                    },
                    {
                        "subject_id": 2,
                        "hadm_id": 200,
                        "time": t0 + timedelta(hours=72),
                        "code": "HOSPITAL_DISCHARGE//DIED",
                    },
                    {
                        "subject_id": 2,
                        "hadm_id": None,
                        "time": t0 + timedelta(hours=70),
                        "code": "MEDS_DEATH",
                    },
                    # hadm 300 (subject 3): two ICU segments separated by floor time
                    {
                        "subject_id": 3,
                        "hadm_id": 300,
                        "time": t0,
                        "code": "HOSPITAL_ADMISSION//EMERGENCY",
                    },
                    {
                        "subject_id": 3,
                        "hadm_id": 300,
                        "time": t0 + timedelta(hours=60),
                        "code": "HOSPITAL_DISCHARGE//HOME",
                    },
                    {
                        "subject_id": 3,
                        "hadm_id": 300,
                        "time": t0 + timedelta(hours=6),
                        "code": "ICU_ADMISSION//MICU",
                    },
                    {
                        "subject_id": 3,
                        "hadm_id": 300,
                        "time": t0 + timedelta(hours=10),
                        "code": "ICU_DISCHARGE//MICU",
                    },
                    {
                        "subject_id": 3,
                        "hadm_id": 300,
                        "time": t0 + timedelta(hours=30),
                        "code": "ICU_ADMISSION//SICU",
                    },
                    {
                        "subject_id": 3,
                        "hadm_id": 300,
                        "time": t0 + timedelta(hours=36),
                        "code": "ICU_DISCHARGE//SICU",
                    },
                ]
            )
            meds.write_parquet(meds_events_dir / "train" / "0.parquet")

            tokens = pl.DataFrame(
                [
                    {
                        "hadm_id": 100,
                        "tokens": [1, 2, 3],
                        "times": [t0, t0 + timedelta(hours=1), t0 + timedelta(hours=2)],
                    },
                    {
                        "hadm_id": 200,
                        "tokens": [4, 5],
                        "times": [t0, t0 + timedelta(hours=2)],
                    },
                    {
                        "hadm_id": 300,
                        "tokens": [6, 7, 8],
                        "times": [t0, t0 + timedelta(hours=3), t0 + timedelta(hours=4)],
                    },
                ]
            )
            tokens.write_parquet(tokenized_dir / "train" / "tokens_timelines.parquet")

            rc = extract_main(
                [
                    "--meds_events_dir",
                    str(meds_events_dir),
                    "--tokenized_dir",
                    str(tokenized_dir),
                    "--splits",
                    "train",
                    "--imv_itemids",
                    "225792,224385",
                    "--los_days",
                    "7",
                    "--observation_hours",
                    "24",
                ]
            )
            self.assertEqual(rc, 0)

            out = pl.read_parquet(tokenized_dir / "train" / "tokens_timelines_outcomes.parquet").sort(
                "hospitalization_id"
            )

            self.assertEqual(out["hospitalization_id"].to_list(), ["100", "200", "300"])

            row100 = out.filter(pl.col("hospitalization_id") == "100").row(0, named=True)
            self.assertFalse(row100["same_admission_death"])
            self.assertFalse(row100["long_length_of_stay"])
            self.assertAlmostEqual(float(row100["icu_length_of_stay"]), 18.0, places=3)
            self.assertTrue(row100["icu_admission"])
            self.assertFalse(row100["icu_admission_24h"])
            self.assertTrue(row100["imv_event"])
            self.assertTrue(row100["imv_event_24h"])
            self.assertFalse(row100["prolonged_icu_stay"])

            row200 = out.filter(pl.col("hospitalization_id") == "200").row(0, named=True)
            self.assertTrue(row200["same_admission_death"])
            self.assertAlmostEqual(float(row200["icu_length_of_stay"]), 0.0, places=3)
            self.assertFalse(row200["icu_admission"])
            self.assertFalse(row200["imv_event"])
            self.assertFalse(row200["prolonged_icu_stay"])

            row300 = out.filter(pl.col("hospitalization_id") == "300").row(0, named=True)
            self.assertFalse(row300["same_admission_death"])
            self.assertAlmostEqual(float(row300["icu_length_of_stay"]), 10.0, places=3)
            self.assertTrue(row300["icu_admission"])
            self.assertTrue(row300["icu_admission_24h"])
            self.assertFalse(row300["imv_event"])
            self.assertFalse(row300["prolonged_icu_stay"])


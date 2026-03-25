from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from scripts.validate_imv_detection import main as validate_main


def _write_outcomes(base: Path, split: str, rows: list[dict]) -> None:
    (base / split).mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(base / split / "tokens_timelines_outcomes.parquet")


class TestValidateImvDetection(unittest.TestCase):
    def test_validate_imv_detection_smoke(self) -> None:
        with TemporaryDirectory() as d:
            tmp_path = Path(d)
            meds_dir = tmp_path / "meds"
            clif_dir = tmp_path / "clif"
            out_json = tmp_path / "report.json"

            # Two hospitalizations, one positive IMV.
            clif_rows = [
                {
                    "hospitalization_id": "100",
                    "imv_event": True,
                    "imv_event_24h": False,
                    "icu_admission": True,
                    "icu_admission_24h": False,
                    "same_admission_death": False,
                    "long_length_of_stay": False,
                },
                {
                    "hospitalization_id": "200",
                    "imv_event": False,
                    "imv_event_24h": False,
                    "icu_admission": False,
                    "icu_admission_24h": False,
                    "same_admission_death": True,
                    "long_length_of_stay": True,
                },
            ]
            meds_rows = [
                # Match CLIF for 100 except imv_event_24h (intentional mismatch)
                {
                    "hospitalization_id": "100",
                    "imv_event": True,
                    "imv_event_24h": True,
                    "icu_admission": True,
                    "icu_admission_24h": False,
                    "same_admission_death": False,
                    "long_length_of_stay": False,
                },
                {
                    "hospitalization_id": "200",
                    "imv_event": False,
                    "imv_event_24h": False,
                    "icu_admission": False,
                    "icu_admission_24h": False,
                    "same_admission_death": True,
                    "long_length_of_stay": True,
                },
            ]

            _write_outcomes(clif_dir, "train", clif_rows)
            _write_outcomes(meds_dir, "train", meds_rows)

            rc = validate_main(
                [
                    "--meds_dir",
                    str(meds_dir),
                    "--clif_dir",
                    str(clif_dir),
                    "--splits",
                    "train",
                    "--output_json",
                    str(out_json),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_json.exists())

            report = json.loads(out_json.read_text())
            self.assertFalse(report["by_split"]["train"]["skipped"])
            self.assertEqual(report["by_split"]["train"]["n_overlap"], 2)

            # imv_event should match perfectly: 1 TP, 1 TN
            imv_counts = report["by_split"]["train"]["labels"]["imv_event"]["counts"]
            self.assertEqual(imv_counts, {"tp": 1, "tn": 1, "fp": 0, "fn": 0})

            # imv_event_24h mismatch for hospitalization_id=100 => 1 FP
            imv24_counts = report["by_split"]["train"]["labels"]["imv_event_24h"]["counts"]
            self.assertEqual(imv24_counts["fp"], 1)


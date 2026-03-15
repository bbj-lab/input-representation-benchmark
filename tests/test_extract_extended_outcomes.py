from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import polars as pl

from scripts.extract_extended_outcomes import _compute_extended_outcomes


class TestExtractExtendedOutcomes(unittest.TestCase):
    def test_post24h_targets_and_first24h_exclusions(self) -> None:
        t0 = datetime(2020, 1, 1, 0, 0, 0)

        meds = pl.DataFrame(
            [
                # hadm 100
                {"subject_id": 1, "hadm_id": 100, "time": t0, "code": "HOSPITAL_ADMISSION//EMERGENCY", "numeric_value": None},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=60), "code": "HOSPITAL_DISCHARGE//HOME", "numeric_value": None},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=10), "code": "LAB//50971//mEq/L", "numeric_value": 6.8},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=30), "code": "LAB//50971//mEq/L", "numeric_value": 5.2},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=8), "code": "LAB//51222//g/dL", "numeric_value": 6.5},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=30), "code": "LAB//50931//mg/dL", "numeric_value": 45.0},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=12), "code": "LAB//50912//mg/dL", "numeric_value": 1.1},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=30), "code": "LAB//50912//mg/dL", "numeric_value": 2.3},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=40), "code": "LAB//50963//pg/mL", "numeric_value": 100.0},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=8), "code": "LAB//50983//mEq/L", "numeric_value": 118.0},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=30), "code": "LAB//50983//mEq/L", "numeric_value": 122.0},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=8), "code": "VITAL//220045//bpm", "numeric_value": 131.0},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=30), "code": "VITAL//220045//bpm", "numeric_value": 100.0},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=30), "code": "VITAL//220052//mmHg", "numeric_value": 60.0},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=8), "code": "INFUSION_START//221906", "numeric_value": 0.1},
                {"subject_id": 1, "hadm_id": 100, "time": t0 + timedelta(hours=8), "code": "PROCEDURE//225441", "numeric_value": None},
                # hadm 200
                {"subject_id": 2, "hadm_id": 200, "time": t0, "code": "HOSPITAL_ADMISSION//EMERGENCY", "numeric_value": None},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=72), "code": "HOSPITAL_DISCHARGE//HOME", "numeric_value": None},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=30), "code": "LAB//50811//g/dL", "numeric_value": 8.0},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=30), "code": "LAB//50983//mEq/L", "numeric_value": 162.0},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=30), "code": "LAB//50971//mEq/L", "numeric_value": 2.3},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=10), "code": "VITAL//220052//mmHg", "numeric_value": 70.0},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=40), "code": "VITAL//220052//mmHg", "numeric_value": 68.0},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=40), "code": "VITAL//220050//mmHg", "numeric_value": 95.0},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=40), "code": "VITAL//220045//bpm", "numeric_value": 135.0},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=40), "code": "VITAL//220179//mmHg", "numeric_value": 190.0},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=30), "code": "VITAL//227290//", "numeric_value": None},
                {"subject_id": 2, "hadm_id": 200, "time": t0 + timedelta(hours=30), "code": "INFUSION_START//221289", "numeric_value": 0.2},
                # hadm 300
                {"subject_id": 3, "hadm_id": 300, "time": t0, "code": "HOSPITAL_ADMISSION//EMERGENCY", "numeric_value": None},
                {"subject_id": 3, "hadm_id": 300, "time": t0 + timedelta(hours=72), "code": "HOSPITAL_DISCHARGE//HOME", "numeric_value": None},
                {"subject_id": 3, "hadm_id": 300, "time": t0 + timedelta(hours=30), "code": "PROCEDURE//225441", "numeric_value": None},
                {"subject_id": 3, "hadm_id": 300, "time": t0 + timedelta(hours=30), "code": "PROCEDURE//225802", "numeric_value": None},
            ]
        )

        out = (
            _compute_extended_outcomes(meds.lazy(), observation_hours=24.0)
            .collect()
            .sort("hospitalization_id")
        )

        self.assertNotIn("time_to_icu_hours", out.columns)

        row100 = out.filter(pl.col("hospitalization_id") == "100").row(0, named=True)
        self.assertAlmostEqual(float(row100["peak_creatinine"]), 2.3, places=6)
        self.assertAlmostEqual(float(row100["peak_potassium"]), 5.2, places=6)
        self.assertEqual(float(row100["hyperkalemia"]), 0.0)
        self.assertTrue(row100["hyperkalemia_24h"])
        self.assertEqual(float(row100["severe_hypokalemia"]), 0.0)
        self.assertFalse(row100["severe_hypokalemia_24h"])
        self.assertIsNone(row100["min_hemoglobin"])
        self.assertIsNone(row100["severe_anemia"])
        self.assertTrue(row100["severe_anemia_24h"])
        self.assertAlmostEqual(float(row100["min_glucose"]), 45.0, places=6)
        self.assertEqual(float(row100["hypoglycemia"]), 1.0)
        self.assertFalse(row100["hypoglycemia_24h"])
        self.assertAlmostEqual(float(row100["min_sodium"]), 122.0, places=6)
        self.assertEqual(float(row100["profound_hyponatremia"]), 1.0)
        self.assertTrue(row100["profound_hyponatremia_24h"])
        self.assertEqual(float(row100["severe_hypernatremia"]), 0.0)
        self.assertFalse(row100["severe_hypernatremia_24h"])
        self.assertEqual(float(row100["tachycardia_hr130"]), 0.0)
        self.assertTrue(row100["tachycardia_hr130_24h"])
        self.assertEqual(float(row100["vasopressor_initiation"]), 0.0)
        self.assertTrue(row100["vasopressor_initiation_24h"])
        self.assertEqual(float(row100["hypotension"]), 1.0)
        self.assertFalse(row100["hypotension_24h"])
        self.assertIsNone(row100["severe_hypertension"])
        self.assertFalse(row100["severe_hypertension_24h"])
        self.assertEqual(float(row100["hemodialysis_initiation"]), 0.0)
        self.assertTrue(row100["hemodialysis_initiation_24h"])
        self.assertEqual(float(row100["crrt_initiation"]), 0.0)
        self.assertFalse(row100["crrt_initiation_24h"])

        row200 = out.filter(pl.col("hospitalization_id") == "200").row(0, named=True)
        self.assertAlmostEqual(float(row200["peak_potassium"]), 2.3, places=6)
        self.assertEqual(float(row200["hyperkalemia"]), 0.0)
        self.assertFalse(row200["hyperkalemia_24h"])
        self.assertEqual(float(row200["severe_hypokalemia"]), 1.0)
        self.assertFalse(row200["severe_hypokalemia_24h"])
        self.assertAlmostEqual(float(row200["min_hemoglobin"]), 8.0, places=6)
        self.assertEqual(float(row200["severe_anemia"]), 0.0)
        self.assertFalse(row200["severe_anemia_24h"])
        self.assertIsNone(row200["min_glucose"])
        self.assertIsNone(row200["hypoglycemia"])
        self.assertFalse(row200["hypoglycemia_24h"])
        self.assertAlmostEqual(float(row200["max_sodium"]), 162.0, places=6)
        self.assertEqual(float(row200["severe_hypernatremia"]), 1.0)
        self.assertFalse(row200["severe_hypernatremia_24h"])
        self.assertEqual(float(row200["profound_hyponatremia"]), 0.0)
        self.assertFalse(row200["profound_hyponatremia_24h"])
        self.assertEqual(float(row200["tachycardia_hr130"]), 1.0)
        self.assertFalse(row200["tachycardia_hr130_24h"])
        self.assertEqual(float(row200["vasopressor_initiation"]), 1.0)
        self.assertFalse(row200["vasopressor_initiation_24h"])
        self.assertEqual(float(row200["hypotension"]), 0.0)
        self.assertFalse(row200["hypotension_24h"])
        self.assertEqual(float(row200["severe_hypertension"]), 1.0)
        self.assertFalse(row200["severe_hypertension_24h"])
        self.assertEqual(float(row200["crrt_initiation"]), 1.0)
        self.assertFalse(row200["crrt_initiation_24h"])
        self.assertEqual(float(row200["hemodialysis_initiation"]), 0.0)
        self.assertFalse(row200["hemodialysis_initiation_24h"])

        row300 = out.filter(pl.col("hospitalization_id") == "300").row(0, named=True)
        self.assertEqual(float(row300["hemodialysis_initiation"]), 1.0)
        self.assertFalse(row300["hemodialysis_initiation_24h"])
        self.assertEqual(float(row300["crrt_initiation"]), 1.0)
        self.assertFalse(row300["crrt_initiation_24h"])


if __name__ == "__main__":
    unittest.main()

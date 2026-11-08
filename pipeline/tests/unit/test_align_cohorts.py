from pipeline.scripts.align_cohorts import extract_icu_cohort


def test_hicu_patients_are_derived_from_same_qualifying_admission() -> None:
    patients, hospitalizations = extract_icu_cohort(
        long_stay_patients={1, 2},
        long_stay_hadm={10, 20},
        icu_patients={1, 2},
        icu_hadm={11, 20},
        subject_to_hadm={1: {10, 11}, 2: {20}},
    )

    # Patient 1 has an ICU stay and a long stay, but on separate admissions.
    assert hospitalizations == {20}
    assert patients == {2}

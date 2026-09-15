"""Tests for period labelling and cycle building, on made-up data."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data


def diary(rows, interval=2022, phase=None):
    """Build a small diary from (id, day, flow_volume) triples."""
    df = pd.DataFrame(rows, columns=["id", "day_in_study", "flow_volume"])
    df["study_interval"] = interval
    if phase is not None:
        df["phase"] = phase
    return df


def test_spotting_is_not_a_period_start():
    df = diary([(1, 1, "Spotting / Very Light"), (1, 2, "Not at all"), (1, 3, "Moderate")])
    assert data.period_starts(df).start_day.tolist() == [3]


def test_missing_flow_is_not_bleeding():
    df = diary([(1, 1, np.nan), (1, 2, "Heavy")])
    assert data.period_starts(df).start_day.tolist() == [2]


def test_consecutive_bleed_days_are_one_period():
    df = diary([(1, d, "Moderate") for d in (1, 2, 3, 4)])
    assert data.period_starts(df).start_day.tolist() == [1]


def test_gap_days_controls_splitting():
    """gap_days is the difference in day_in_study between bleeding days."""
    df = diary([(1, d, "Light") for d in (1, 2, 6, 7)])
    assert data.period_starts(df, gap_days=3).start_day.tolist() == [1, 6]
    assert data.period_starts(df, gap_days=4).start_day.tolist() == [1]


def test_two_unlogged_days_do_not_split_a_period():
    """The reason the default is 3: a couple of missed logs are not a new period."""
    df = diary([(1, 1, "Heavy"), (1, 2, np.nan), (1, 3, np.nan), (1, 4, "Moderate")])
    assert data.period_starts(df, gap_days=3).start_day.tolist() == [1]


def test_interval_2_labels_come_from_phase():
    """Interval 2 reported no flow, so its bleeding days come from phase.

    Starts shift by PHASE_START_OFFSET because phase turns Menstrual a day
    before any flow is logged.
    """
    df = diary([(1, d, np.nan) for d in (838, 839, 866)], interval=2024,
               phase=["Menstrual", "Menstrual", "Menstrual"])
    assert data.period_starts(df).start_day.tolist() == [839, 867]


def test_phase_offset_does_not_change_cycle_length():
    """A uniform shift moves both ends, so the baselines see an identical problem."""
    df = diary([(1, d, np.nan) for d in (838, 866, 894)], interval=2024,
               phase=["Menstrual"] * 3)
    assert data.build_cycles(data.period_starts(df)).cycle_length.tolist() == [28, 28]


def test_phase_is_ignored_when_flow_was_reported():
    """Interval 1 reported flow, so phase must not add extra starts."""
    df = diary([(1, 1, "Heavy"), (1, 10, "Not at all")], phase=["Menstrual", "Menstrual"])
    assert data.period_starts(df).start_day.tolist() == [1]


def test_last_run_has_no_cycle():
    df = diary([(1, 1, "Heavy"), (1, 29, "Heavy"), (1, 57, "Heavy")])
    starts = data.period_starts(df)
    cycles = data.build_cycles(starts)
    assert len(starts) == 3
    assert cycles.cycle_length.tolist() == [28, 28]


def test_implausible_lengths_dropped():
    df = diary([(1, 1, "Heavy"), (1, 6, "Heavy"), (1, 41, "Heavy")])
    cycles = data.build_cycles(data.period_starts(df, gap_days=3), min_len=15, max_len=60)
    assert cycles.cycle_length.tolist() == [35]


def test_target_counts_down():
    cycles = pd.DataFrame([{"id": 1, "study_interval": 2022, "cycle_index": 0,
                            "start_day": 10, "next_start_day": 38, "cycle_length": 28}])
    f = data.make_daily_prediction_frame(cycles)
    assert f.cycle_day.tolist() == list(range(1, 29))
    assert f.days_until_next_period.iloc[0] == 28
    assert f.days_until_next_period.iloc[-1] == 1
    assert (f.days_until_next_period == f.cycle_length - f.cycle_day + 1).all()


def test_cycle_day_resets_and_never_tracks_day_in_study():
    cycles = pd.DataFrame([
        {"id": 1, "study_interval": 2022, "cycle_index": 0, "start_day": 1,
         "next_start_day": 29, "cycle_length": 28},
        {"id": 1, "study_interval": 2022, "cycle_index": 1, "start_day": 29,
         "next_start_day": 57, "cycle_length": 28},
    ])
    f = data.make_daily_prediction_frame(cycles)
    assert f.cycle_day.max() == 28
    assert f.day_in_study.max() == 56


def test_split_keeps_participants_on_one_side():
    df = pd.DataFrame({"id": np.repeat(np.arange(1, 21), 3),
                       "study_interval": [2022, 2024, 2022] * 20})
    s = data.participant_split(df, test_frac=0.3, seed=1)
    assert not set(s.train_ids) & set(s.test_ids)
    assert len(s.train_ids) + len(s.test_ids) == 20
    tr, te = s.apply(df)
    assert not set(tr.id) & set(te.id)

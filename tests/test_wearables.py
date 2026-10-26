"""Tests for the trailing wearable features, on made-up data."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import wearables as wear


def signals(rows, interval=2022):
    """Build a daily signal frame from (id, day, resting_hr) triples."""
    df = pd.DataFrame(rows, columns=["id", "day_in_study", "resting_hr"])
    df["study_interval"] = interval
    for col in wear.SIGNALS:
        if col not in df:
            df[col] = np.nan
    return df.sort_values(wear.KEY).reset_index(drop=True)


def steady(pid, n, value=60.0, start=1):
    return [(pid, d, value) for d in range(start, start + n)]


def test_dense_days_fills_gaps_between_readings():
    df = signals([(1, 1, 60.0), (1, 5, 62.0)])
    out = wear.dense_days(df)
    assert out.day_in_study.tolist() == [1, 2, 3, 4, 5]
    assert out.resting_hr.isna().sum() == 3


def test_dense_days_does_not_pad_past_the_last_reading():
    df = signals(steady(1, 3) + steady(2, 10))
    out = wear.dense_days(df)
    assert (out.id == 1).sum() == 3


def test_baseline_excludes_today():
    # Ten days at 60 then one at 70: the deviation is the full 10, not a blended value.
    df = signals(steady(1, 10) + [(1, 11, 70.0)])
    out = wear.trailing_features(df, window=14, min_readings=8)
    assert out.loc[out.day_in_study == 11, "resting_hr_dev"].item() == 10.0


def test_features_are_nan_until_the_window_has_enough_readings():
    df = signals(steady(1, 12))
    out = wear.trailing_features(df, window=14, min_readings=8)
    # Day 9 is the first with 8 prior readings behind it.
    got = out.set_index("day_in_study").resting_hr_dev
    assert got.loc[:8].isna().all()
    assert got.loc[9:].notna().all()


def test_a_gap_does_not_let_the_window_reach_further_back():
    # One reading, a 20-day gap, then 8 more: the window never holds 8 until the run does.
    df = signals([(1, 1, 60.0)] + steady(1, 8, start=21))
    out = wear.trailing_features(df, window=14, min_readings=8)
    assert out.resting_hr_dev.notna().sum() == 0


def test_window_does_not_reach_across_participants():
    df = signals(steady(1, 20, value=60.0) + steady(2, 20, value=90.0))
    out = wear.trailing_features(df, window=14, min_readings=8)
    second = out[out.id == 2].set_index("day_in_study")
    # If participant 1 leaked in, participant 2 would start with a large deviation.
    assert second.resting_hr_dev.loc[:8].isna().all()
    assert (second.resting_hr_dev.dropna() == 0).all()


def test_window_does_not_reach_across_intervals_for_one_participant():
    df = pd.concat([signals(steady(1, 20, value=60.0), interval=2022),
                    signals(steady(1, 20, value=90.0), interval=2024)], ignore_index=True)
    out = wear.trailing_features(df, window=14, min_readings=8)
    later = out[out.study_interval == 2024].set_index("day_in_study")
    assert (later.resting_hr_dev.dropna() == 0).all()


def test_z_is_the_deviation_over_the_trailing_spread():
    values = [60.0, 62.0, 58.0, 61.0, 59.0, 63.0, 57.0, 60.0, 70.0]
    df = signals([(1, d + 1, v) for d, v in enumerate(values)])
    out = wear.trailing_features(df, window=14, min_readings=8)
    row = out[out.day_in_study == 9].iloc[0]
    prior = pd.Series(values[:8])
    assert np.isclose(row.resting_hr_dev, 70.0 - prior.median())
    assert np.isclose(row.resting_hr_z, row.resting_hr_dev / prior.std())


def test_flat_history_gives_no_z():
    df = signals(steady(1, 12))
    out = wear.trailing_features(df, window=14, min_readings=8)
    assert out.resting_hr_z.isna().all()


def test_add_features_keeps_every_row_of_the_frame():
    frame = pd.DataFrame({"id": [1, 1, 9], "study_interval": 2022,
                          "day_in_study": [10, 11, 10], "days_until_next_period": [2, 1, 7]})
    out = wear.add_features(frame, wear.trailing_features(signals(steady(1, 20))))
    assert len(out) == 3
    # Participant 9 has no wearable record, so the join leaves the features empty.
    assert out.loc[out.id == 1, "resting_hr_dev"].notna().all()
    assert out.loc[out.id == 9, "resting_hr_dev"].isna().all()


def test_feature_columns_are_the_eight_names():
    cols = wear.feature_columns()
    assert len(cols) == 8
    assert cols[:2] == ["resting_hr_dev", "resting_hr_z"]

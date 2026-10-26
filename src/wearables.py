"""Daily wearable signals and trailing-baseline features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import data_path

KEY = ["id", "study_interval", "day_in_study"]
PARTICIPANT = ["id", "study_interval"]

SIGNALS = ("resting_hr", "skin_temp", "hrv_rmssd", "breathing_rate")

# TODO(TIMING_ALIGNMENT): This is a heuristic, not a documented dataset rule.
# Verify HRV epoch dates against sleep sessions before using wake-day model features.
# Currently, readings from this hour onward are assigned to the following day.
EVENING_HOUR = 18

# Provisional RMSSD upper threshold; validate before modelling.
RMSSD_MAX = 300


def _resting_hr(data_dir):
    df = pd.read_csv(data_path("resting_heart_rate.csv", data_dir))
    # Exclude nonpositive HR values from daily aggregation.
    df = df[df.value > 0]
    # TODO(INTERVAL2_REVIEW): Revisit aggregation and repeated value/error pairs
    # after the dataset authors respond, before including Interval 2 in modelling.
    # The existing daily minimum is provisional, not a validated correction.
    return df.groupby(KEY, as_index=False).value.min().rename(columns={"value": "resting_hr"})


def _skin_temp(data_dir):
    df = pd.read_csv(data_path("computed_temperature.csv", data_dir),
                     usecols=PARTICIPANT + ["sleep_end_day_in_study", "nightly_temperature"])
    df = df.rename(columns={"sleep_end_day_in_study": "day_in_study"})
    # Aggregate sleep sessions ending on the same day.
    return df.groupby(KEY, as_index=False).nightly_temperature.mean().rename(
        columns={"nightly_temperature": "skin_temp"})


def _hrv(data_dir):
    df = pd.read_csv(data_path("heart_rate_variability_details.csv", data_dir),
                     usecols=KEY + ["timestamp", "rmssd"])
    df = df[(df.rmssd > 0) & (df.rmssd < RMSSD_MAX)]
    evening = pd.to_datetime(df.timestamp).dt.hour >= EVENING_HOUR
    df = df.assign(day_in_study=df.day_in_study + evening.astype(int))
    # Aggregate the assigned night's readings into one daily mean.
    return df.groupby(KEY, as_index=False).rmssd.mean().rename(columns={"rmssd": "hrv_rmssd"})


def _breathing_rate(data_dir):
    df = pd.read_csv(data_path("respiratory_rate_summary.csv", data_dir),
                     usecols=KEY + ["full_sleep_breathing_rate"])
    df = df[df.full_sleep_breathing_rate > 0]
    return df.groupby(KEY, as_index=False).full_sleep_breathing_rate.mean().rename(
        columns={"full_sleep_breathing_rate": "breathing_rate"})


def load_daily_signals(data_dir=None):
    """One row per participant-day, with a column per signal in SIGNALS."""
    frames = [_resting_hr(data_dir), _skin_temp(data_dir), _hrv(data_dir),
              _breathing_rate(data_dir)]
    daily = frames[0]
    for f in frames[1:]:
        daily = daily.merge(f, on=KEY, how="outer")
    return daily.sort_values(KEY).reset_index(drop=True)


def dense_days(daily):
    """Insert the missing days between each participant's first and last reading.

    The trailing window counts calendar days, so a gap has to be present as a row of
    NaN rather than absent. Without this a window reaches further back than it claims.
    """
    spans = daily.groupby(PARTICIPANT).day_in_study.agg(["min", "max"]).reset_index()
    grid = pd.concat(
        [pd.DataFrame({"id": r.id, "study_interval": r.study_interval,
                       "day_in_study": np.arange(r.min, r.max + 1)})
         for r in spans.itertuples(index=False)],
        ignore_index=True)
    return grid.merge(daily, on=KEY, how="left").sort_values(KEY).reset_index(drop=True)


def trailing_features(daily, window=14, min_readings=8):
    """Each signal against the participant's own preceding days.

    Adds <signal>_dev, today minus the trailing median in the signal's own units, and
    <signal>_z, that deviation over the trailing standard deviation. Both are NaN until
    the window holds min_readings actual readings.
    """
    out = dense_days(daily)

    # shift(1) drops today, so nothing in the baseline comes from the day being predicted.
    # The rolling has to stay inside transform or it runs over the participant boundary.
    def trail(s, how):
        return getattr(s.shift(1).rolling(window, min_periods=min_readings), how)()

    for col in SIGNALS:
        grp = out.groupby(PARTICIPANT)[col]
        centre = grp.transform(trail, "median")
        spread = grp.transform(trail, "std")
        out[f"{col}_dev"] = out[col] - centre
        out[f"{col}_z"] = out[f"{col}_dev"] / spread.replace(0, np.nan)
    return out


def feature_columns(signals=SIGNALS):
    """The 8 trailing feature names, in SIGNALS order."""
    return [f"{s}_{kind}" for s in signals for kind in ("dev", "z")]


def add_features(frame, features):
    """Left-join the trailing features onto the daily prediction frame."""
    cols = KEY + feature_columns()
    return frame.merge(features[cols], on=KEY, how="left")

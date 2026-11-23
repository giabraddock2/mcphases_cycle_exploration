"""Calendar-only baselines for predicting days until the next period."""
from __future__ import annotations

import numpy as np
import pandas as pd

# A prediction of 0 is the floor: today or overdue.
MIN_PREDICTION = 0.0


class PopulationBaseline:
    """One typical cycle length for everyone, taken from the training cycles."""

    def __init__(self, statistic="median"):
        self.statistic = statistic
        self.cycle_length_ = None

    def fit(self, train_cycles):
        """Median length of the training cycles, one value per cycle."""
        self.cycle_length_ = float(getattr(train_cycles.cycle_length, self.statistic)())
        return self

    def predict(self, frame):
        # days_until = cycle_length - cycle_day + 1, so predicting days remaining
        # is the same as estimating the current cycle's length.
        return np.clip(self.cycle_length_ - frame.cycle_day.to_numpy() + 1, MIN_PREDICTION, None)


class PersonalBaseline:
    """Earlier cycle lengths within the same participant and study interval."""

    def __init__(self, statistic="median", fallback=None):
        self.statistic = statistic
        self.fallback = fallback
        self.history_ = {}

    def fit(self, train_cycles):
        """Store [cycle_index, length] arrays per participant and interval.

        Cycle indices restart in each interval, so they cannot order history
        across intervals. Reset history between intervals, matching the newer
        history-feature models; do not implicitly carry prior-study history.
        """
        self.history_ = {
            key: g.sort_values("cycle_index")[["cycle_index", "cycle_length"]].to_numpy()
            for key, g in train_cycles.groupby(["id", "study_interval"])
        }
        if self.fallback is not None and self.fallback.cycle_length_ is None:
            self.fallback.fit(train_cycles)
        return self

    def _estimate(self, pid, study_interval, cycle_index):
        """Earlier cycles in this participant-interval, or the fallback."""
        hist = self.history_.get((pid, study_interval))
        if hist is not None:
            # column 0 is cycle_index, column 1 is length; keep rows before this cycle
            prior = hist[hist[:, 0] < cycle_index][:, 1]
            if len(prior):
                stat = np.median(prior) if self.statistic == "median" else np.mean(prior)
                return float(stat), False
        # No earlier cycle, so use the cohort estimate. It is fitted on training
        # participants only, so nothing from the test set reaches it.
        return float(self.fallback.cycle_length_), True

    def predict(self, frame, return_fallback_mask=False):
        est = np.empty(len(frame))
        used_fb = np.zeros(len(frame), dtype=bool)
        for i, (pid, interval, k) in enumerate(zip(
            frame.id.to_numpy(), frame.study_interval.to_numpy(), frame.cycle_index.to_numpy()
        )):
            est[i], used_fb[i] = self._estimate(pid, interval, int(k))
        pred = np.clip(est - frame.cycle_day.to_numpy() + 1, MIN_PREDICTION, None)
        return (pred, used_fb) if return_fallback_mask else pred


class CalendarDayModel:
    """Average days remaining at each cycle day, learned from the training rows.

    Assumes no fixed cycle length, so a cycle that has already run long is not
    treated as overdue.
    """

    def __init__(self):
        self.table_ = {}
        self.global_mean_ = None

    def fit(self, train_frame):
        """Average days remaining for each cycle day, e.g. {1: 28.4, 2: 27.5, ...}."""
        self.global_mean_ = float(train_frame.days_until_next_period.mean())
        grp = train_frame.groupby("cycle_day").days_until_next_period.agg(["mean", "size"])
        # Late cycle days have few rows, so their average is noisy. Days with
        # fewer than 5 fall through to the overall mean.
        self.table_ = {int(d): float(r["mean"]) for d, r in grp.iterrows() if r["size"] >= 5}
        return self

    def predict(self, frame):
        # Look up each row's cycle day; unseen or sparse days get the overall mean.
        vals = [self.table_.get(int(d), self.global_mean_) for d in frame.cycle_day]
        return np.clip(np.asarray(vals, dtype=float), MIN_PREDICTION, None)


def predict_expanding_personal(frame, cycles, fallback, statistic="median"):
    """Personalised predictions for participants already seen.

    cycles is the full cycles DataFrame. Each cycle is predicted from the median
    of that participant's earlier cycles in the same interval only. History
    resets between intervals. Returns (predictions, fallback mask).
    """
    model = PersonalBaseline(statistic=statistic, fallback=fallback).fit(cycles)
    return model.predict(frame, return_fallback_mask=True)

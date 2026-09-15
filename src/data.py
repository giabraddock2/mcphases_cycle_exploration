"""Loading, period labelling and cycle building for the mcPHASES diary."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Flow levels counted as a period day. Spotting and "Not at all" are not.
BLEED_LEVELS = ("Light", "Somewhat Light", "Moderate", "Somewhat Heavy", "Heavy", "Very Heavy")

# Interval 2 has no flow_volume at all, so its labels come from phase instead.
PHASE_BLEED = "Menstrual"

# Adjust phase start to after spotting days (92 of 117 Interval 1 cycles).
PHASE_START_OFFSET = 1

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
)


def data_path(filename, data_dir=None):
    """Path to a file in the local data directory."""
    return os.path.join(data_dir or DEFAULT_DATA_DIR, filename)


def load_diary(data_dir=None, interval=None):
    """Load the daily diary. Pass interval=2022 or 2024 for one interval only."""
    df = pd.read_csv(data_path("hormones_and_selfreport.csv", data_dir))
    if interval is not None:
        df = df[df.study_interval == interval]
    return df.sort_values(["id", "study_interval", "day_in_study"]).reset_index(drop=True)


def check_duplicates(df):
    """Check no participant has two rows for the same study day.

    A repeated day would count one bleeding report twice and could invent a
    period start.
    """
    keys = [
        ("id + study_interval + day_in_study", ["id", "study_interval", "day_in_study"]),
        ("id + day_in_study", ["id", "day_in_study"]),
    ]
    return pd.DataFrame(
        [{"key": k, "duplicate_rows": int(df.duplicated(c).sum())} for k, c in keys]
    )


def missingness(df):
    """Per-column dtype and missing percentage."""
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (100 * df.isna().mean()).round(2),
    })
    return out.sort_values("pct_missing", ascending=False)


def coverage_summary(df):
    """Participants, rows and day range per study interval."""
    rows = []
    for interval, g in df.groupby("study_interval"):
        rows.append({
            "study_interval": interval,
            "participants": g.id.nunique(),
            "rows": len(g),
            "day_min": int(g.day_in_study.min()),
            "day_max": int(g.day_in_study.max()),
            "days_per_participant_median": int(g.groupby("id").day_in_study.nunique().median()),
            "flow_reported_pct": round(100 * g.flow_volume.notna().mean(), 1),
        })
    return pd.DataFrame(rows)


def flow_value_counts(df):
    """Reported flow levels, flagged by whether each counts as a period day."""
    out = df.flow_volume.value_counts(dropna=False).rename_axis("flow_volume").reset_index(name="n_days")
    out["counts_as_period_day"] = out.flow_volume.isin(BLEED_LEVELS)
    out["pct_of_days"] = (100 * out.n_days / len(df)).round(2)
    return out


def add_bleed_flag(df):
    """Mark bleeding days, from flow_volume where reported and phase otherwise."""
    bleed = df.flow_volume.isin(BLEED_LEVELS)
    if "phase" in df.columns:
        no_flow = df.groupby("study_interval").flow_volume.transform(lambda s: s.notna().sum() == 0)
        bleed = bleed | (no_flow & (df.phase == PHASE_BLEED))
    return df.assign(is_bleed=bleed)


def bleed_days(g):
    """Days this participant bled.

    The offset applies only where phase supplied the days, which is Interval 2.
    Interval 1 reported flow directly and is left alone.
    """
    days = np.sort(g.loc[g.is_bleed, "day_in_study"].unique())
    if g.flow_volume.notna().sum() == 0:
        return days + PHASE_START_OFFSET
    return days


def period_starts(df, gap_days=3):
    """Find period start days: a bleeding day more than gap_days after the last one.

    gap_days is an assumption this modelling makes. At 3, bleeding on days 1, 2, 6
    splits into two periods and days 1, 2, 5 stay as one. Set to 3 rather than 2 so
    a couple of unlogged days mid-period do not read as a new period.
    """
    df = add_bleed_flag(df)
    rows = []
    for (pid, interval), g in df.groupby(["id", "study_interval"]):
        days = bleed_days(g)
        if len(days) == 0:
            continue
        starts = [int(days[0])] + [int(d) for prev, d in zip(days, days[1:]) if d - prev > gap_days]
        rows.extend({"id": pid, "study_interval": interval, "start_day": s} for s in starts)
    return pd.DataFrame(rows, columns=["id", "study_interval", "start_day"])


def build_cycles(starts, min_len=15, max_len=60):
    """Pair consecutive starts into completed cycles.

    Bounds are deliberately wider than the clinical normal range of 21-35 days.
    Roughly a third of this cohort reported irregular periods, and tightening to
    21-35 drops 12 cycles and 7 of the participants who have enough history for a
    personalised model.
    """
    rows = []
    for (pid, interval), g in starts.groupby(["id", "study_interval"]):
        s = np.sort(g.start_day.unique())
        # Pairing drops each participant's last run, which has no next start.
        for k, (a, b) in enumerate(zip(s, s[1:])):
            rows.append({"id": pid, "study_interval": interval, "cycle_index": k,
                         "start_day": int(a), "next_start_day": int(b), "cycle_length": int(b - a)})
    cols = ["id", "study_interval", "cycle_index", "start_day", "next_start_day", "cycle_length"]
    cycles = pd.DataFrame(rows, columns=cols)
    if len(cycles):
        cycles = cycles[cycles.cycle_length.between(min_len, max_len)].reset_index(drop=True)
    return cycles


def censoring_report(starts, cycles, min_len=15, max_len=60):
    """Where period starts went: how many became usable cycles and what dropped."""
    n_starts = len(starts)
    n_pairs = n_starts - starts.groupby(["id", "study_interval"]).ngroups
    return pd.DataFrame([
        {"stage": "period starts found", "n": n_starts},
        {"stage": "last run per participant, no next start", "n": n_starts - n_pairs},
        {"stage": "consecutive start pairs", "n": n_pairs},
        {"stage": f"dropped, length outside {min_len}-{max_len} days", "n": n_pairs - len(cycles)},
        {"stage": "usable cycles", "n": len(cycles)},
    ])


def cycles_per_participant(cycles):
    """How many participants have 1, 2, 3... usable cycles."""
    return (cycles.groupby("id").size().value_counts()
            .rename_axis("cycles_per_participant").reset_index(name="n_participants")
            .sort_values("cycles_per_participant").reset_index(drop=True))


def make_daily_prediction_frame(cycles):
    """One row per participant-day, with days_until_next_period as the target.

    cycle_day is 1 on the first day of flow and counts up; the target counts down
    to 1 on the day before the next period. Both reset at every period, so
    cycle_day stays within one cycle and never tracks day_in_study.
    """
    frames = []
    for r in cycles.itertuples(index=False):
        days = np.arange(r.start_day, r.next_start_day)
        frames.append(pd.DataFrame({
            "id": r.id, "study_interval": r.study_interval, "cycle_index": r.cycle_index,
            "day_in_study": days, "cycle_day": days - r.start_day + 1,
            "cycle_length": r.cycle_length, "days_until_next_period": r.next_start_day - days,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@dataclass
class ParticipantSplit:
    """Train and test participant ids, with no overlap."""

    train_ids: np.ndarray
    test_ids: np.ndarray

    def apply(self, df):
        return df[df.id.isin(self.train_ids)].copy(), df[df.id.isin(self.test_ids)].copy()


def participant_split(df, test_frac=0.3, seed=0):
    """Hold out whole participants, grouping on id across both intervals.

    The 20 interval 2 participants all returned from interval 1, so splitting by
    row or by interval would put the same person on both sides.
    """
    ids = np.sort(df.id.unique())
    shuffled = np.random.default_rng(seed).permutation(ids)
    n_test = max(1, int(round(test_frac * len(ids))))
    return ParticipantSplit(np.sort(shuffled[n_test:]), np.sort(shuffled[:n_test]))

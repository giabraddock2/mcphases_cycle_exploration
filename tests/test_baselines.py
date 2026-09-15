"""Tests for the baselines, mainly that none of them sees the future."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import baselines as base
from src import data
from src import evaluation as evals


def make_cycles(id_lengths):
    """Build a cycles frame from {participant_id: [length, length, ...]}."""
    rows = []
    for pid, lens in id_lengths.items():
        day = 1
        for k, L in enumerate(lens):
            rows.append({"id": pid, "study_interval": 2022, "cycle_index": k,
                         "start_day": day, "next_start_day": day + L, "cycle_length": L})
            day += L
    return pd.DataFrame(rows)


def test_population_uses_training_median():
    m = base.PopulationBaseline().fit(make_cycles({1: [26, 28], 2: [30, 32]}))
    assert m.cycle_length_ == 29.0
    assert np.allclose(m.predict(pd.DataFrame({"cycle_day": [1, 10, 29]})), [29, 20, 1])


def test_predictions_never_go_negative():
    m = base.PopulationBaseline().fit(make_cycles({1: [28, 28]}))
    assert (m.predict(pd.DataFrame({"cycle_day": [50, 90]})) >= 0).all()


def test_personal_ignores_the_current_and_later_cycles():
    """Cycle 0 is 20 days and cycle 1 is 40. Predicting cycle 1 may only see 20."""
    c = make_cycles({1: [20, 40]})
    m = base.PersonalBaseline(fallback=base.PopulationBaseline().fit(c)).fit(c)
    est, used_fallback = m._estimate(1, 1)
    assert est == 20.0
    assert not used_fallback


def test_first_cycle_uses_the_fallback():
    c = make_cycles({1: [20, 40], 2: [30, 30]})
    pop = base.PopulationBaseline().fit(c)
    est, used_fallback = base.PersonalBaseline(fallback=pop).fit(c)._estimate(1, 0)
    assert used_fallback
    assert est == pop.cycle_length_


def test_unknown_participant_uses_the_fallback():
    c = make_cycles({1: [28, 28]})
    pop = base.PopulationBaseline().fit(c)
    est, used_fallback = base.PersonalBaseline(fallback=pop).fit(c)._estimate(999, 3)
    assert used_fallback and est == pop.cycle_length_


def test_calendar_day_learns_the_average_at_each_day():
    """Six cycles, so cycle day 1 clears the five-row threshold."""
    c = make_cycles({1: [28, 28, 28], 2: [28, 28, 28]})
    m = base.CalendarDayModel().fit(data.make_daily_prediction_frame(c))
    assert np.isclose(m.table_[1], 28.0)
    assert np.isclose(m.predict(pd.DataFrame({"cycle_day": [1]}))[0], 28.0)


def test_calendar_day_falls_back_on_an_unseen_day():
    c = make_cycles({1: [28, 28, 28], 2: [28, 28, 28]})
    m = base.CalendarDayModel().fit(data.make_daily_prediction_frame(c))
    assert np.isclose(m.predict(pd.DataFrame({"cycle_day": [999]}))[0], m.global_mean_)


def test_calendar_day_skips_days_with_too_few_rows():
    """Only two cycles reach day 40, so it falls through to the overall mean."""
    c = make_cycles({1: [28, 45], 2: [28, 45]})
    m = base.CalendarDayModel().fit(data.make_daily_prediction_frame(c))
    assert 40 not in m.table_


def test_a_perfect_prediction_scores_zero():
    f = data.make_daily_prediction_frame(make_cycles({1: [28, 28]}))
    assert evals.mae(f.days_until_next_period, f.days_until_next_period) == 0.0


def test_interval_brackets_the_mae():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": np.repeat(np.arange(12), 20)})
    df["y"] = rng.normal(14, 5, len(df))
    df["p"] = df.y + rng.normal(0, 2, len(df))
    out = evals.mae_interval(df, "y", "p", n_boot=300, seed=0)
    assert out["ci_low"] <= out["MAE"] <= out["ci_high"]
    assert out["n_participants"] == 12

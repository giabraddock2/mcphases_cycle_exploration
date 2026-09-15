"""Error metrics, uncertainty and plots.

Functions here summarise across participants, so nothing they return can be
traced back to one person's daily records.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    """Mean absolute error: average of |predicted - actual|, in days.

    An MAE of 3 means predictions miss the next period by about 3 days.
    """
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def error_summary(y_true, y_pred):
    """Headline error numbers for one model.

    MAE is the average miss and median_AE the typical miss, which ignores a few
    bad cycles. bias is signed, so a negative value means predicting too early.
    within_1d and friends are the share of days landing that close.
    """
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    a = np.abs(err)
    return {
        "n": int(len(err)),
        "MAE": round(float(a.mean()), 3),
        "median_AE": round(float(np.median(a)), 3),
        "bias": round(float(err.mean()), 3),
        "within_1d": round(100 * float((a <= 1).mean()), 1),
        "within_2d": round(100 * float((a <= 2).mean()), 1),
        "within_3d": round(100 * float((a <= 3).mean()), 1),
    }


def per_participant_mae(df, y_col, pred_col):
    """One MAE per participant."""
    out = (df.assign(_ae=(df[pred_col] - df[y_col]).abs())
             .groupby("id").agg(n_days=("_ae", "size"), MAE=("_ae", "mean"))
             .round({"MAE": 3}).reset_index())
    return out.sort_values("MAE").reset_index(drop=True)


def mae_interval(df, y_col, pred_col, n_boot=2000, seed=0, alpha=0.05):
    """MAE after resampling, and the middle 95% of those values.

    The train/test split is fixed here; only the scored test rows are redrawn.
    n_boot is how many redraws, alpha sets the interval width. Participants are
    drawn whole, not row by row, since days within a cycle share one error.
    """
    rng = np.random.default_rng(seed)
    groups = [g[[y_col, pred_col]].to_numpy() for _, g in df.groupby("id")]
    scores = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        arr = np.concatenate([groups[i] for i in pick])
        scores[b] = np.mean(np.abs(arr[:, 1] - arr[:, 0]))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"MAE": round(mae(df[y_col], df[pred_col]), 3), "ci_low": round(float(lo), 3),
            "ci_high": round(float(hi), 3), "n_participants": len(groups), "n_rows": len(df)}


def repeated_split_scores(cycles, frame, fit_predict, n_splits=20, test_frac=0.3):
    """MAE across many random participant splits, not one.

    fit_predict takes (train_cycles, train_frame, test_cycles, test_frame) and
    returns {model name: predictions} for the test frame.
    """
    from . import data as _data

    rows = []
    for seed in range(n_splits):
        split = _data.participant_split(cycles, test_frac=test_frac, seed=seed)
        trc, tec = split.apply(cycles)
        trf, tef = split.apply(frame)
        for name, pred in fit_predict(trc, trf, tec, tef).items():
            rows.append({"seed": seed, "model": name,
                         "MAE": mae(tef.days_until_next_period, pred)})
    return pd.DataFrame(rows)


def summarise_splits(scores):
    """Mean, spread and range of MAE across the splits, one row per model."""
    out = (scores.groupby("model").MAE
           .agg(n_splits="size", mean_MAE="mean", sd="std", best="min", worst="max")
           .round(3).reset_index())
    return out.sort_values("mean_MAE").reset_index(drop=True)


def compare_models(df, y_col, pred_cols, n_boot=2000, seed=0):
    """One row per model, with the MAE interval alongside the point estimate."""
    rows = []
    for label, col in pred_cols.items():
        s = error_summary(df[y_col], df[col])
        b = mae_interval(df, y_col, col, n_boot=n_boot, seed=seed)
        s.update({"model": label, "MAE_95CI": f"[{b['ci_low']}, {b['ci_high']}]"})
        rows.append(s)
    cols = ["model", "n", "MAE", "MAE_95CI", "median_AE", "bias", "within_1d", "within_2d", "within_3d"]
    return pd.DataFrame(rows)[cols]


def plot_pred_vs_actual(ax, y_true, y_pred, title, max_day=None):
    """Density plot of predicted against actual days remaining."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    hi = max_day or int(max(y_true.max(), y_pred.max())) + 1
    hb = ax.hexbin(y_true, y_pred, gridsize=28, cmap="Blues", mincnt=1, extent=(0, hi, 0, hi))
    ax.plot([0, hi], [0, hi], color="#c2185b", lw=1.2, ls="--", label="perfect")
    ax.set_xlabel("actual days until next period")
    ax.set_ylabel("predicted")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.legend(fontsize=8, loc="upper left")
    return hb


def plot_mae_by_cycle_day(ax, df, y_col, pred_cols, max_day=35):
    """Error against how far into the cycle the prediction was made."""
    for label, col in pred_cols.items():
        g = (df.assign(_ae=(df[col] - df[y_col]).abs())
               .query("cycle_day <= @max_day").groupby("cycle_day")._ae.mean())
        ax.plot(g.index, g.values, lw=1.6, marker="o", ms=3, label=label)
    ax.set_xlabel("cycle day")
    ax.set_ylabel("MAE (days)")
    ax.set_title("Error by position in cycle", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

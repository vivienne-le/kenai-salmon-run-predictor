"""XGBoost models, evaluation, and an optional Prophet baseline."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from . import config as C
from .features import feature_cols

PARAMS = dict(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=3, random_state=42)


def _fit(X, y):
    m = XGBRegressor(**PARAMS)
    m.fit(X, np.log1p(y))
    return m


def _predict(m, X):
    return np.clip(np.expm1(m.predict(X)), 0, None)


def season_metrics(df: pd.DataFrame) -> dict:
    """Daily MAE/RMSE plus season-level peak-date, peak-volume and total-run errors."""
    per_year = []
    for y, g in df.groupby("year"):
        a, p = g.loc[g["count"].idxmax()], g.loc[g["pred"].idxmax()]
        per_year.append({"year": int(y),
                         "peak_date_actual": str(a.date.date()), "peak_date_pred": str(p.date.date()),
                         "peak_day_error": int(p.doy - a.doy),
                         "peak_count_actual": float(a["count"]), "peak_count_pred": float(g["pred"].max()),
                         "total_actual": float(g["count"].sum()), "total_pred": float(g["pred"].sum())})
    py = pd.DataFrame(per_year)
    return {"daily_MAE": float(mean_absolute_error(df["count"], df["pred"])),
            "daily_RMSE": float(np.sqrt(mean_squared_error(df["count"], df["pred"]))),
            "peak_date_MAE_days": float(py.peak_day_error.abs().mean()),
            "peak_volume_MAPE_pct": float((abs(py.peak_count_pred - py.peak_count_actual) / py.peak_count_actual).mean() * 100),
            "total_run_MAPE_pct": float((abs(py.total_pred - py.total_actual) / py.total_actual).mean() * 100),
            "per_year": per_year}


def train_season(df: pd.DataFrame):
    """Leave-one-year-out CV for the preseason model, then fit on all years."""
    cols = feature_cols(df)
    oof = []
    for y in sorted(df.year.unique()):
        tr, te = df[df.year != y], df[df.year == y].copy()
        te["pred"] = _predict(_fit(tr[cols], tr["count"]), te[cols])
        oof.append(te)
    oof = pd.concat(oof)
    return _fit(df[cols], df["count"]), cols, oof, season_metrics(oof)


def train_nowcast(df: pd.DataFrame, holdout_years: int = 5):
    """Next-day in-season model, tested on the most recent seasons."""
    cols = feature_cols(df)
    cut = sorted(df.year.unique())[-holdout_years]
    tr, te = df[df.year < cut], df[df.year >= cut].copy()
    te["pred"] = _predict(_fit(tr[cols], tr["count"]), te[cols])
    model = _fit(df[cols], df["count"])
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    return model, cols, te, season_metrics(te), imp


def prophet_baseline(counts: pd.DataFrame, holdout_years: int = 5):
    """Optional Prophet yearly-seasonality baseline (pip install prophet)."""
    try:
        from prophet import Prophet
    except ImportError:
        return None
    d = counts.rename(columns={"date": "ds", "count": "y"})
    d["y"] = np.log1p(d["y"])
    cut = sorted(d.ds.dt.year.unique())[-holdout_years]
    tr, te = d[d.ds.dt.year < cut], d[d.ds.dt.year >= cut].copy()
    m = Prophet(yearly_seasonality=20, weekly_seasonality=False, daily_seasonality=False).fit(tr)
    te["pred"] = np.clip(np.expm1(m.predict(te[["ds"]])["yhat"].values), 0, None)
    te = te.rename(columns={"ds": "date"}); te["count"] = np.expm1(te["y"])
    te["year"] = te.date.dt.year; te["doy"] = te.date.dt.dayofyear
    return season_metrics(te)


def save_json(obj, name):
    (C.OUTPUTS / name).write_text(json.dumps(obj, indent=2))

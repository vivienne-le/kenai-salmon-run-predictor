"""Feature engineering: lag and rolling features, seasonal markers, preseason summaries."""
from __future__ import annotations
import numpy as np
import pandas as pd

ENV_COLS = ["air_temp_c", "water_temp_c", "discharge_cfs", "precip_mm", "snowfall_cm", "sst_c"]


def _window_mean(env, year, col, m1, d1, m2, d2):
    s = env.loc[(env.date >= pd.Timestamp(year, m1, d1)) & (env.date <= pd.Timestamp(year, m2, d2)), col]
    return s.mean() if col in env and len(s) else np.nan


def preseason_summary(env: pd.DataFrame, years) -> pd.DataFrame:
    """One row per year with conditions known before the run starts (by July 1).
    This is what lets the model forecast a season ahead, e.g. 2026."""
    rows = []
    for y in years:
        r = {"year": y}
        if "air_temp_c" in env: r["pre_air_temp"] = _window_mean(env, y, "air_temp_c", 5, 1, 6, 30)
        if "snowfall_cm" in env:
            w = env[(env.date >= pd.Timestamp(y - 1, 11, 1)) & (env.date <= pd.Timestamp(y, 4, 30))]
            r["winter_snow"] = w["snowfall_cm"].sum() if len(w) > 150 else np.nan
        if "precip_mm" in env:
            w = env[(env.date >= pd.Timestamp(y, 5, 1)) & (env.date <= pd.Timestamp(y, 6, 30))]
            r["pre_precip"] = w["precip_mm"].sum() if len(w) else np.nan
        if "discharge_cfs" in env: r["pre_discharge"] = _window_mean(env, y, "discharge_cfs", 6, 1, 6, 30)
        if "discharge_cfs_kenai_cooper_landing" in env:
            r["pre_discharge_upper"] = _window_mean(env, y, "discharge_cfs_kenai_cooper_landing", 6, 1, 6, 30)
        if "water_temp_c" in env: r["pre_water_temp"] = _window_mean(env, y, "water_temp_c", 6, 1, 6, 30)
        if "sst_c" in env: r["pre_sst"] = _window_mean(env, y, "sst_c", 4, 1, 6, 30)
        rows.append(r)
    return pd.DataFrame(rows)


def season_frame(counts: pd.DataFrame, pre: pd.DataFrame, years=None) -> pd.DataFrame:
    """Daily rows for the season model: day of year plus preseason conditions.
    `counts` may be None (forecasting): then only the calendar is built."""
    if counts is not None:
        df = counts.copy()
    else:
        from . import config as C
        lo, hi = C.SEASON_DOY
        if C.RUN == "late":
            lo = max(lo, C.LATE_RUN_START_DOY)
        df = pd.concat([pd.DataFrame({"date": pd.date_range(pd.Timestamp(y, 1, 1) + pd.Timedelta(days=lo - 1),
                                                           pd.Timestamp(y, 1, 1) + pd.Timedelta(days=hi - 1))})
                        for y in years])
    df["year"] = df.date.dt.year
    df["doy"] = df.date.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df.doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.doy / 365.25)
    df = df.merge(pre, on="year", how="left")
    return df


def nowcast_frame(counts: pd.DataFrame, env: pd.DataFrame | None = None) -> pd.DataFrame:
    """In-season features: 1-7 day count lags, rolling stats, and (if daily data is available) telemetry."""
    df = (counts.merge(env, on="date", how="left") if env is not None else counts.copy()).sort_values("date")
    df["year"] = df.date.dt.year
    df["doy"] = df.date.dt.dayofyear
    g = df.groupby("year")["count"]
    for k in range(1, 8):
        df[f"lag_{k}"] = g.shift(k)
    shifted = g.shift(1)
    df["roll_mean_3"] = shifted.groupby(df.year).transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["roll_mean_7"] = shifted.groupby(df.year).transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["roll_std_7"] = shifted.groupby(df.year).transform(lambda s: s.rolling(7, min_periods=2).std())
    df["cum_to_date"] = shifted.groupby(df.year).cumsum()
    for c in [c for c in ENV_COLS if c in df]:
        df[f"{c}_lag1"] = df.groupby("year")[c].shift(1)
        df[f"{c}_roll3"] = df.groupby("year")[c].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return df.dropna(subset=["lag_1"])


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"date", "count", "year", "pred"}
    return [c for c in df.columns if c not in drop]

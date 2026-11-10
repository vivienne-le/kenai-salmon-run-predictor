import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import config as C

FORECAST, ACTUAL, MUTED = "#2f5aa8", "#d65f6f", "#8a8a8a"  # validated for colour-blind separation


def seasonal_curves(counts: pd.DataFrame):
    df = counts.assign(year=counts.date.dt.year, doy=counts.date.dt.dayofyear)
    years = sorted(df.year.unique())
    cmap = plt.get_cmap("viridis", len(years))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, y in enumerate(years):
        g = df[df.year == y]
        ax.plot(g.doy, g["count"].rolling(3, center=True, min_periods=1).mean(), color=cmap(i), lw=1, alpha=.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(years[0], years[-1]))
    fig.colorbar(sm, ax=ax, label="Year")
    ax.set(xlabel="Day of year", ylabel="Daily count (3-day mean)", title="Sockeye migration by season")
    fig.tight_layout(); fig.savefig(C.OUTPUTS / "seasonal_curves.png", dpi=150); plt.close(fig)


def peak_shift(counts: pd.DataFrame, pre: pd.DataFrame):
    """Two stacked panels sharing the year axis (never a dual y-axis)."""
    df = counts.assign(year=counts.date.dt.year, doy=counts.date.dt.dayofyear)
    peaks = df.loc[df.groupby("year")["count"].idxmax(), ["year", "doy"]]
    col = "pre_air_temp"
    has_t = col in pre
    fig, axes = plt.subplots(2 if has_t else 1, 1, figsize=(10, 6 if has_t else 3.5), sharex=True, squeeze=False)
    ax1 = axes[0, 0]
    z = np.polyfit(peaks.year, peaks.doy, 1)
    ax1.plot(peaks.year, peaks.doy, "o-", color=FORECAST, lw=2, ms=5)
    ax1.plot(peaks.year, np.polyval(z, peaks.year), "--", color=MUTED, lw=1.5)
    ax1.set(ylabel="Peak day of year", title=f"Peak migration date ({z[0]*10:+.1f} days per decade)")
    if has_t:
        t = pre.set_index("year")[col]
        t = t[t.index.isin(peaks.year)]
        ax2 = axes[1, 0]
        ax2.plot(t.index, t.values, "o-", color=ACTUAL, lw=2, ms=5)
        ax2.set(ylabel="Air temp, May-Jun (°C)", title="Preseason air temperature, Soldotna")
    axes[-1, 0].set_xlabel("Year")
    for ax in axes.flat:
        ax.grid(axis="y", color="#e6e6e3"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(C.OUTPUTS / "peak_shift.png", dpi=150); plt.close(fig)


def forecast_plot(fc: pd.DataFrame, counts: pd.DataFrame, actual: pd.DataFrame | None = None):
    hist = counts.assign(doy=counts.date.dt.dayofyear).groupby("doy")["count"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(hist.quantile(.1).index, hist.quantile(.1), hist.quantile(.9), color="grey", alpha=.2,
                    label="Historical 10-90%")
    ax.plot(fc.doy, fc.pred, color=FORECAST, lw=2, label=f"{C.FORECAST_YEAR} forecast")
    if actual is not None and len(actual):
        ax.plot(actual.date.dt.dayofyear, actual["count"], color=ACTUAL, lw=1.5, label=f"{C.FORECAST_YEAR} actual")
    ax.set(xlabel="Day of year", ylabel="Daily count", title=f"{C.FORECAST_YEAR} forecast" + (" vs actual" if actual is not None else ""))
    ax.legend(); fig.tight_layout()
    name = "forecast_vs_actual_2026.png" if actual is not None else "forecast_2026.png"
    fig.savefig(C.OUTPUTS / name, dpi=150); plt.close(fig)


def oof_plot(oof: pd.DataFrame, title: str, fname: str):
    years = sorted(oof.year.unique())[-6:]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for ax, y in zip(axes.flat, years):
        g = oof[oof.year == y]
        ax.plot(g.doy, g["count"], color=MUTED, label="Actual")
        ax.plot(g.doy, g["pred"], color=FORECAST, label="Predicted")
        ax.set_title(str(y))
    axes.flat[0].legend(); fig.suptitle(title); fig.tight_layout()
    fig.savefig(C.OUTPUTS / fname, dpi=150); plt.close(fig)

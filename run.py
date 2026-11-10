"""Salmon Run Predictor. Usage:

  python run.py synth      # write synthetic test data (offline)
  python run.py fetch-env  # download USGS + Open-Meteo + NOAA SST telemetry
  python run.py train      # cross-validate and train both XGBoost models
  python run.py forecast   # preseason forecast for the 2026 run
  python run.py compare    # compare the 2026 forecast with actual 2026 counts
  python run.py dashboard  # build outputs/dashboard.html (open it in a browser)
  python run.py site       # build outputs/site/index.html for web hosting
"""
import sys
import pandas as pd
from salmon import config as C, data, features as F, models as M, plots as P


def _load():
    counts = data.load_fish_counts()
    return counts[counts.date.dt.year <= C.END_YEAR], data.load_preseason(), data.load_environment()


def train():
    counts, pre, env = _load()
    print(f"{counts.date.dt.year.nunique()} seasons, {len(counts)} daily rows")
    P.seasonal_curves(counts); P.peak_shift(counts, pre)

    sdf = F.season_frame(counts, pre)
    model, cols, oof, met = M.train_season(sdf)
    model.save_model(C.OUTPUTS / "season_model.json")
    M.save_json({"features": cols, "metrics": met}, "season_metrics.json")
    P.oof_plot(oof, "Preseason model, leave-one-year-out", "season_cv.png")
    print("Preseason model (leave-one-year-out):", {k: round(v, 2) for k, v in met.items() if k != "per_year"})

    ndf = F.nowcast_frame(counts, env)
    nmodel, ncols, te, nmet, imp = M.train_nowcast(ndf)
    nmodel.save_model(C.OUTPUTS / "nowcast_model.json")
    M.save_json({"features": ncols, "metrics": nmet, "importance": imp.round(4).to_dict()}, "nowcast_metrics.json")
    P.oof_plot(te, "In-season next-day model, recent holdout seasons", "nowcast_holdout.png")
    print("Next-day model (holdout):", {k: round(v, 2) for k, v in nmet.items() if k != "per_year"})
    print("Top features:", ", ".join(imp.index[:6]))

    base = M.prophet_baseline(counts)
    if base:
        M.save_json(base, "prophet_baseline.json")
        print("Prophet baseline:", {k: round(v, 2) for k, v in base.items() if k != "per_year"})


def forecast():
    from xgboost import XGBRegressor
    import json, numpy as np
    counts, pre, _ = _load()
    meta = json.loads((C.OUTPUTS / "season_metrics.json").read_text())
    m = XGBRegressor(); m.load_model(C.OUTPUTS / "season_model.json")
    fc = F.season_frame(None, pre, years=[C.FORECAST_YEAR])
    missing = [c for c in meta["features"] if fc[c].isna().all()]
    if missing:
        print("WARNING: no preseason data for", missing, "- forecast falls back on the model's defaults")
    fc["pred"] = np.clip(np.expm1(m.predict(fc[meta["features"]])), 0, None)
    fc[["date", "doy", "pred"]].to_csv(C.OUTPUTS / "forecast_2026.csv", index=False)
    pk = fc.loc[fc.pred.idxmax()]
    print(f"{C.FORECAST_YEAR} forecast: peak {pk.date.date()} (~{pk.pred:,.0f} fish/day), "
          f"season total ~{fc.pred.sum():,.0f}")
    print(f"Typical peak-date error from cross-validation: +/-{meta['metrics']['peak_date_MAE_days']:.1f} days")
    P.forecast_plot(fc, counts)


def compare():
    fc = pd.read_csv(C.OUTPUTS / "forecast_2026.csv", parse_dates=["date"])
    allc = data.load_fish_counts()
    act = allc[allc.date.dt.year == C.FORECAST_YEAR]
    if act["count"].sum() == 0:
        sys.exit(f"No {C.FORECAST_YEAR} counts in {C.FISH_COUNTS_CSV}. Add the 2026 ADF&G export first.")
    df = act.merge(fc[["date", "pred"]], on="date")
    df["year"] = C.FORECAST_YEAR; df["doy"] = df.date.dt.dayofyear
    met = M.season_metrics(df)
    M.save_json(met, "forecast_vs_actual_2026.json")
    r = met["per_year"][0]
    print(f"Peak: forecast {r['peak_date_pred']} vs actual {r['peak_date_actual']} ({r['peak_day_error']:+d} days)")
    print(f"Total: forecast {r['total_pred']:,.0f} vs actual {r['total_actual']:,.0f}")
    print(f"Daily MAE {met['daily_MAE']:,.0f}, RMSE {met['daily_RMSE']:,.0f}")
    P.forecast_plot(fc, allc[allc.date.dt.year <= C.END_YEAR], actual=act)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    C.RAW.mkdir(parents=True, exist_ok=True); C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    {"synth": data.make_synthetic, "fetch-env": data.fetch_environment, "train": train,
     "forecast": forecast, "compare": compare,
     "dashboard": lambda: __import__("salmon.export_dashboard", fromlist=["build"]).build(),
     "site": lambda: __import__("salmon.export_dashboard", fromlist=["build"]).build(web=True)}.get(cmd, lambda: print(__doc__))()

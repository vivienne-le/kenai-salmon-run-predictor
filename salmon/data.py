"""Load fish counts and fetch environmental telemetry."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import requests
from . import config as C


# fish counts
def load_fish_counts(path=C.FISH_COUNTS_CSV) -> pd.DataFrame:
    """Read an ADF&G Fish Counts export.

    Expected columns:
    year, count_date, fish_count, Species, Species_ID, count_location, count_location_ID
    Returns a daily frame with columns: date, count.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "species" in df.columns:
        df = df[df["species"].astype(str).str.contains(C.SPECIES, case=False)]
    df["date"] = pd.to_datetime(df["count_date"], format="mixed").dt.normalize()
    df = df.groupby("date", as_index=False)["fish_count"].sum().rename(columns={"fish_count": "count"})
    doy = df["date"].dt.dayofyear
    if C.RUN == "late":
        df = df[doy >= C.LATE_RUN_START_DOY]
    elif C.RUN == "early":
        df = df[doy < C.LATE_RUN_START_DOY]
    return _fill_season(df)


def _fill_season(df: pd.DataFrame) -> pd.DataFrame:
    """Add zero-count days so every season covers the same day-of-year window."""
    lo, hi = C.SEASON_DOY
    if C.RUN == "late":
        lo = max(lo, C.LATE_RUN_START_DOY)
    frames = []
    for yr in sorted(df["date"].dt.year.unique()):
        days = pd.date_range(pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=lo - 1),
                             pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=hi - 1))
        frames.append(pd.DataFrame({"date": days}))
    full = pd.concat(frames)
    return full.merge(df, on="date", how="left").fillna({"count": 0})


# environment
def fetch_usgs(start: str, end: str) -> pd.DataFrame:
    out = []
    for i, (name, site) in enumerate(C.USGS_SITES.items()):
        suffix = "" if i == 0 else f"_{name}"  # first gauge feeds the main features
        r = requests.get("https://waterservices.usgs.gov/nwis/dv/", timeout=60, params={
            "format": "json", "sites": site, "startDT": start, "endDT": end,
            "parameterCd": ",".join(C.USGS_PARAMS)})
        r.raise_for_status()
        for ts in r.json()["value"]["timeSeries"]:
            code = ts["variable"]["variableCode"][0]["value"]
            vals = ts["values"][0]["value"]
            s = pd.DataFrame(vals)
            if s.empty:
                continue
            s = pd.DataFrame({"date": pd.to_datetime(s["dateTime"]).dt.tz_localize(None).dt.normalize(),
                              f"{C.USGS_PARAMS[code]}{suffix}": pd.to_numeric(s["value"], errors="coerce")})
            out.append(s.set_index("date"))
    return pd.concat(out, axis=1).groupby(level=0).mean().reset_index() if out else pd.DataFrame(columns=["date"])


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params={
        "latitude": C.LAT, "longitude": C.LON, "start_date": start, "end_date": end,
        "daily": "temperature_2m_mean,precipitation_sum,snowfall_sum", "timezone": "America/Anchorage"})
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({"date": pd.to_datetime(d["time"]), "air_temp_c": d["temperature_2m_mean"],
                         "precip_mm": d["precipitation_sum"], "snowfall_cm": d["snowfall_sum"]})


def fetch_sst(start: str, end: str) -> pd.DataFrame:
    """NOAA OISST v2.1 daily SST at one ocean grid point via ERDDAP."""
    url = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv?"
           f"sst[({start}T12:00:00Z):1:({end}T12:00:00Z)][(0.0)][({C.SST_LAT})][({C.SST_LON})]")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
    return pd.DataFrame({"date": pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize(),
                         "sst_c": df["sst"]})


def fetch_environment(start_year=C.START_YEAR, end_year=C.FORECAST_YEAR) -> pd.DataFrame:
    start, end = f"{start_year}-01-01", f"{end_year}-12-31"
    env = pd.DataFrame({"date": pd.date_range(start, end)})
    for name, fn in [("USGS", fetch_usgs), ("Open-Meteo", fetch_weather), ("SST", fetch_sst)]:
        try:
            env = env.merge(fn(start, end), on="date", how="left")
            print(f"  fetched {name}")
        except Exception as e:  # one source being down shouldn't stop the rest
            print(f"  WARNING: {name} failed: {e}")
    env.to_csv(C.RAW / "environment.csv", index=False)
    write_preseason(env)
    return env


def write_preseason(env: pd.DataFrame) -> pd.DataFrame:
    from .features import preseason_summary
    pre = preseason_summary(env, range(C.START_YEAR, C.FORECAST_YEAR + 1))
    pre.round(3).to_csv(C.PRESEASON_CSV, index=False)
    return pre


def load_environment() -> pd.DataFrame | None:
    """Daily telemetry, if it has been downloaded (only the in-season model uses it)."""
    p = C.RAW / "environment.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def load_preseason() -> pd.DataFrame:
    return pd.read_csv(C.PRESEASON_CSV)


# synthetic
def make_synthetic(seed: int = 7) -> None:
    """Write synthetic data so the pipeline can be tested offline.
    Peak timing shifts earlier in warmer years."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(f"{C.START_YEAR}-01-01", f"{C.FORECAST_YEAR}-12-31")
    doy = days.dayofyear.values
    yr = days.year.values
    warm = {y: rng.normal(0, 1) + 0.04 * (y - C.START_YEAR) for y in np.unique(yr)}
    w = np.array([warm[y] for y in yr])
    env = pd.DataFrame({
        "date": days,
        "air_temp_c": 2 + 12 * np.sin((doy - 100) / 365 * 2 * np.pi) + w + rng.normal(0, 2, len(days)),
        "water_temp_c": np.clip(1 + 9 * np.sin((doy - 110) / 365 * 2 * np.pi) + 0.6 * w, 0, None),
        "discharge_cfs": 1500 + 4000 * np.exp(-((doy - 200) / 35) ** 2) * (1 + 0.1 * w) + rng.normal(0, 150, len(days)),
        "precip_mm": rng.gamma(0.6, 3, len(days)),
        "snowfall_cm": np.where(np.abs(doy - 182) > 120, rng.gamma(0.5, 2, len(days)), 0),
        "sst_c": 6 + 5 * np.sin((doy - 140) / 365 * 2 * np.pi) + 0.7 * w,
    })
    env.to_csv(C.RAW / "environment.csv", index=False)
    write_preseason(env)

    rows = []
    for y in range(C.START_YEAR, C.END_YEAR + 1):
        peak = 204 - 2.5 * warm[y] + rng.normal(0, 2)
        total = rng.lognormal(np.log(1.8e6), 0.35)
        for d in range(C.SEASON_DOY[0], C.SEASON_DOY[1] + 1):
            mu = total * np.exp(-0.5 * ((d - peak) / 7) ** 2) / (7 * np.sqrt(2 * np.pi))
            rows.append({"year": y, "count_date": (pd.Timestamp(y, 1, 1) + pd.Timedelta(days=d - 1)).date(),
                         "fish_count": int(rng.poisson(max(mu, 0.1))), "Species": "Sockeye",
                         "Species_ID": C.SPECIES_ID, "count_location": "Kenai River (SYNTHETIC)",
                         "count_location_ID": C.COUNT_LOCATION_ID})
    pd.DataFrame(rows).to_csv(C.FISH_COUNTS_CSV, index=False)
    print("Wrote SYNTHETIC data to data/raw/. Replace with the real ADF&G export before trusting results.")

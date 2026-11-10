"""Bake model outputs into the self-contained HTML dashboard (dashboard/template.html)."""
import base64
import shutil
import json
import pandas as pd
from . import config as C, data

TEMPLATE = C.ROOT / "dashboard" / "template.html"


FONTS = [("Caveat", 400, "caveat-latin-400-normal"), ("Caveat", 600, "caveat-latin-600-normal"),
         ("Inter", 400, "inter-latin-400-normal"), ("Inter", 500, "inter-latin-500-normal"),
         ("Inter", 600, "inter-latin-600-normal")]


def fonts_css() -> str:
    """Inline the fonts (SIL OFL) so the page works offline and embeds anywhere."""
    out = []
    for fam, w, f in FONTS:
        b64 = base64.b64encode((C.ROOT / "dashboard" / "fonts" / f"{f}.woff2").read_bytes()).decode()
        out.append(f"@font-face{{font-family:'{fam}';font-weight:{w};font-display:swap;"
                   f"src:url(data:font/woff2;base64,{b64}) format('woff2')}}")
    return "\n".join(out)


WEB_FONTS = "@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@400;600&family=Inter:wght@400;500;600&display=swap');"


ICONS = C.ROOT / "dashboard" / "icons"
SITE_FILES = ["salmon-icon.webp", "favicon.ico"]


def _data_uri(name: str, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode((ICONS / name).read_bytes()).decode()


def icons_html(web: bool) -> str:
    """Favicon links: separate files for the hosted site, inlined for the offline file."""
    if web:
        return '<link rel="icon" href="favicon.ico" sizes="16x16 32x32">'
    return f'<link rel="icon" type="image/png" sizes="32x32" href="{_data_uri("favicon-32.png")}">'


def build(web: bool = False) -> None:
    """web=False: one offline file with fonts embedded. web=True: lighter site/index.html for hosting."""
    counts = data.load_fish_counts()
    counts["year"], counts["doy"] = counts.date.dt.year, counts.date.dt.dayofyear
    fc = pd.read_csv(C.OUTPUTS / "forecast_2026.csv", parse_dates=["date"])
    meta = json.loads((C.OUTPUTS / "season_metrics.json").read_text())
    raw = pd.read_csv(C.FISH_COUNTS_CSV, usecols=lambda c: c.lower() == "count_location")
    synthetic = raw.iloc[:, 0].astype(str).str.contains("SYNTHETIC").any() if raw.shape[1] else False

    doys = fc.doy.astype(int).tolist()
    start, end = doys[0], doys[-1]
    hist = counts[counts.year <= C.END_YEAR]

    def season(g):
        g = g.set_index("doy").reindex(range(start, end + 1), fill_value=0)
        c = g["count"].astype(float)
        return {"counts": [round(v) for v in c], "peakDoy": int(c.idxmax()), "total": round(c.sum())}

    seasons = [{"year": int(y), **season(g)} for y, g in hist.groupby("year")]
    act = counts[(counts.year == C.FORECAST_YEAR)]
    actual_fc = {"year": C.FORECAST_YEAR, **season(act)} if act["count"].sum() > 0 else None

    piv = hist.pivot_table(index="doy", columns="year", values="count").reindex(range(start, end + 1)).fillna(0)
    payload = {
        "forecastYear": C.FORECAST_YEAR, "startDoy": start, "endDoy": end, "synthetic": bool(synthetic),
        "river": "Kenai River, Alaska", "species": "late-run sockeye",
        "forecast": [round(v) for v in fc.pred], "forecastPeakDoy": int(fc.loc[fc.pred.idxmax(), "doy"]),
        "band": {"lo": [round(v) for v in piv.quantile(.1, axis=1)], "hi": [round(v) for v in piv.quantile(.9, axis=1)]},
        "seasons": seasons, "actualForecastYear": actual_fc,
        "metrics": {"peakDateMAE": round(meta["metrics"]["peak_date_MAE_days"], 1),
                    "totalMAPE": round(meta["metrics"]["total_run_MAPE_pct"], 1)},
        "team": C.TEAM,
        "cv": [{"year": r["year"], "err": r["peak_day_error"]} for r in meta["metrics"]["per_year"]],
    }
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload)).replace("/*__FONTS__*/", WEB_FONTS if web else fonts_css()).replace("<!--__ICONS__-->", icons_html(web))
    html = html.replace("__MASCOT__", "salmon-icon.webp" if web else _data_uri("salmon-icon.png"))
    out = C.OUTPUTS / "site" / "index.html" if web else C.OUTPUTS / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    if web:
        for name in SITE_FILES:
            shutil.copy(ICONS / name, out.parent / name)
    print(f"Wrote {out.relative_to(C.ROOT)}  (open it in a browser)")

"""Lightweight Streamlit view of the model outputs (the main dashboard is dashboard/template.html).

Run:  streamlit run dashboard.py
Needs: data/raw/fish_counts.csv, then `python run.py train` and `python run.py forecast`.
"""
import json
import altair as alt
import pandas as pd
import streamlit as st
from salmon import config as C, data

FORECAST, ACTUAL, BAND, GRID = "#2f5aa8", "#d4677d", "#d9d9d6", "#ececea"  # validated pair

st.set_page_config(page_title="Kenai Sockeye Run Predictor", page_icon="🐟", layout="wide")


@st.cache_data
def load():
    counts = data.load_fish_counts()
    counts["year"], counts["doy"] = counts.date.dt.year, counts.date.dt.dayofyear
    fc = pd.read_csv(C.OUTPUTS / "forecast_2026.csv", parse_dates=["date"])
    meta = json.loads((C.OUTPUTS / "season_metrics.json").read_text())
    return counts, fc, meta


try:
    counts, fc, meta = load()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Add the ADF&G export, then run "
             "`python run.py train` and `python run.py forecast`.")
    st.stop()

hist = counts[counts.year <= C.END_YEAR]
actual_fc_year = counts[(counts.year == C.FORECAST_YEAR) & (counts["count"] > 0)]
met = meta["metrics"]
pk = fc.loc[fc.pred.idxmax()]


def fmt_day(doy):
    return (pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=int(doy) - 1)).strftime("%b %d")


def style(ch, height=320):
    return (ch.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domain=False, tickColor=GRID, labelColor="#52514e",
                            titleColor="#52514e", labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(orient="top", title=None, labelFontSize=12, symbolStrokeWidth=3))


# header
st.title("Kenai River Sockeye Run Predictor")
st.caption("Late-run sockeye salmon, ADF&G sonar counts · XGBoost forecast from preseason river, "
           "weather and ocean conditions")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{C.FORECAST_YEAR} forecast peak", pk.date.strftime("%b %d"))
c2.metric("Peak daily count", f"{pk.pred:,.0f}")
c3.metric("Forecast season total", f"{fc.pred.sum()/1e6:,.2f} M")
c4.metric("Peak-date accuracy (back-tested)", f"±{met['peak_date_MAE_days']:.1f} days")

# season curve
st.subheader("Season curve")
years = sorted(hist.year.unique(), reverse=True)
options = ([C.FORECAST_YEAR] if len(actual_fc_year) else []) + years
pick = st.selectbox("Compare the forecast with season", options, index=0)

band = (hist.groupby("doy")["count"].quantile([.1, .9]).unstack().reset_index()
        .rename(columns={0.1: "lo", 0.9: "hi"}))
comp = counts[counts.year == pick][["doy", "count"]].rename(columns={"count": "Actual"})
wide = fc[["doy", "pred"]].rename(columns={"pred": "Forecast"}).merge(comp, on="doy", how="left").merge(band, on="doy")
wide["day"] = wide.doy.map(fmt_day)
long = wide.melt(id_vars=["doy", "day"], value_vars=["Forecast", "Actual"], var_name="series", value_name="count")
long["series"] = long.series.map({"Forecast": f"{C.FORECAST_YEAR} forecast", "Actual": f"{pick} actual"})
color = alt.Color("series:N", scale=alt.Scale(domain=[f"{C.FORECAST_YEAR} forecast", f"{pick} actual"],
                                              range=[FORECAST, ACTUAL]))
x = alt.X("doy:Q", title=None, scale=alt.Scale(nice=False),
          axis=alt.Axis(grid=False, tickCount=9, labelExpr="timeFormat(datetime(2001, 0, datum.value), '%b %d')"))
hover = alt.selection_point(fields=["doy"], nearest=True, on="pointerover", empty=False)

band_layer = alt.Chart(wide).mark_area(color=BAND, opacity=.6).encode(
    x=x, y=alt.Y("lo:Q", title="Fish per day"), y2="hi:Q")
lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x=x, y="count:Q", color=color)
rule = alt.Chart(wide).mark_rule(color="#8a8a8a").encode(
    x=x, opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    tooltip=[alt.Tooltip("day:N", title="Date"),
             alt.Tooltip("Forecast:Q", format=",.0f", title=f"{C.FORECAST_YEAR} forecast"),
             alt.Tooltip("Actual:Q", format=",.0f", title=f"{pick} actual"),
             alt.Tooltip("lo:Q", format=",.0f", title="Historical 10th pct"),
             alt.Tooltip("hi:Q", format=",.0f", title="Historical 90th pct")]).add_params(hover)
dots = alt.Chart(long).mark_point(filled=True, size=70, stroke="white", strokeWidth=2).encode(
    x=x, y="count:Q", color=color, opacity=alt.condition(hover, alt.value(1), alt.value(0))
).transform_filter(hover)
st.altair_chart(style(band_layer + lines + rule + dots, 360), width="stretch")
st.caption("Grey band: middle 80% of historical daily counts.")

# history
left, right = st.columns(2)
seasons = (hist.groupby("year")
           .apply(lambda g: pd.Series({"peak_doy": g.loc[g["count"].idxmax(), "doy"], "total": g["count"].sum()}),
                  include_groups=False)
           .reset_index())
seasons["peak_day"] = seasons.peak_doy.map(fmt_day)

with left:
    st.subheader("Is the run arriving earlier?")
    base = alt.Chart(seasons).encode(x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0, values=seasons.year[::5].tolist())))
    line = base.mark_line(color=FORECAST, strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True, color=FORECAST)).encode(
        y=alt.Y("peak_doy:Q", title="Peak day of year", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("year:O", title="Season"), alt.Tooltip("peak_day:N", title="Peak")])
    trend = base.transform_regression("year", "peak_doy").mark_line(color="#8a8a8a", strokeDash=[4, 4], strokeWidth=1.5).encode(
        x="year:O", y="peak_doy:Q")
    st.altair_chart(style(line + trend, 260), width="stretch")
    st.caption("Dashed line: linear trend.")

with right:
    st.subheader("Season totals")
    bars = alt.Chart(seasons).mark_bar(color=FORECAST, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0, values=seasons.year[::5].tolist())),
        y=alt.Y("total:Q", title="Fish counted"),
        tooltip=[alt.Tooltip("year:O", title="Season"), alt.Tooltip("total:Q", format=",.0f", title="Total")])
    st.altair_chart(style(bars, 260), width="stretch")

# accuracy
st.subheader("How accurate is the model?")
st.caption("Each season predicted by a model that never saw it (leave-one-year-out).")
acc = pd.DataFrame(met["per_year"])[["year", "peak_date_actual", "peak_date_pred", "peak_day_error",
                                     "total_actual", "total_pred"]]
acc.columns = ["Season", "Actual peak", "Predicted peak", "Peak error (days)", "Actual total", "Predicted total"]
st.dataframe(acc.sort_values("Season", ascending=False), hide_index=True, width="stretch",
             column_config={"Actual total": st.column_config.NumberColumn(format="%d"),
                            "Predicted total": st.column_config.NumberColumn(format="%d")})

with st.expander("Data table: forecast by day"):
    st.dataframe(wide[["day", "Forecast", "Actual", "lo", "hi"]].rename(
        columns={"day": "Date", "lo": "Hist. 10th pct", "hi": "Hist. 90th pct"}).round(0),
        hide_index=True, width="stretch")

st.caption("Data: Alaska Department of Fish and Game sonar counts; USGS NWIS; Open-Meteo; NOAA OISST. "
           "Built by Vivienne Le and Earthan Le.")

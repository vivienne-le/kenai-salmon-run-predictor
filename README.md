# Salmon Run Predictor

**Live dashboard:** https://kenai-salmon-run-predictor.vercel.app/

Forecasts **when the late-run sockeye salmon return to the Kenai River, Alaska, peaks, and how big it will be**, using historical ADF&G sonar counts combined with river, weather and ocean telemetry. The results are shown in an interactive dashboard.

**Team:** [Vivienne Le](https://www.linkedin.com/in/le-vivienne) designed the dashboard and worked on the Python code. [Earthan Le](https://www.linkedin.com/in/earthan-le/) (UCSC, B.S. Electrical Engineering) worked on the research and the Python code.

## Data
| Source | What | How |
|---|---|---|
| Alaska Department of Fish and Game | Daily late-run sockeye sonar counts, Kenai River | CSV export from the [Fish Counts](https://www.adfg.alaska.gov/sf/FishCounts/) portal |
| USGS NWIS | River discharge and water temperature (Soldotna, Cooper Landing) | REST API |
| Open-Meteo | Air temperature, precipitation, snowfall | REST API |
| NOAA OISST (ERDDAP) | Sea-surface temperature, lower Cook Inlet | REST API |

## Results
Back-tested by leaving out one season at a time (47 seasons, 1979–2025), compared with simply predicting the average season curve:

| | Preseason model | Average-curve baseline |
|---|---|---|
| Peak date error | ±6.0 days | ±6.3 days |
| Season total error | 34% | 44% |
| Daily count MAE | 15,600 fish | 17,100 fish |

**2026 check:** the model predicted a peak on **July 24**, and the real run peaked on July 24. It overestimated the season total: it forecast 1.55M, while ADF&G's preliminary count is 1.11M (counting ended Aug 10).

Run timing is only weakly explained by preseason conditions. The model clearly improves on the baseline for run size, but only slightly for the peak date.

## Pipeline
1. **Ingestion** (`salmon/data.py`): load the fish counts and fetch the telemetry.
2. **Features** (`salmon/features.py`):
   - 1–7 day count lags, rolling mean/std, season-to-date totals
   - day-of-year seasonal markers
   - same-day, lagged and rolling environmental readings
   - preseason summaries (May–June air temperature, June water temperature and flow, spring SST, winter snowfall)
3. **Models** (`salmon/models.py`):
   - **Preseason model**, XGBoost: predicts the whole season curve from conditions known before the run starts. This model produces the 2026 forecast. Evaluated with leave-one-year-out cross-validation.
   - **In-season model**, XGBoost: next-day counts from recent counts and telemetry, tested on the most recent seasons.
   - **Prophet baseline** (optional): yearly-seasonality decomposition for comparison.
   - **Metrics:** daily MAE/RMSE, peak-date error (days), peak-volume and total-run % error.
4. **Dashboard** (`dashboard/template.html` → `outputs/dashboard.html`): a single self-contained HTML file with the data and fonts embedded, so it works offline and can be embedded anywhere.
   - A handwritten forecast summary that writes itself out
   - A season curve that draws itself in, for comparing the forecast with any past season
   - A "gooey" dot grid of every season, day by day, with the peaks marked
   - Back-test accuracy and season totals
   - Hover sounds (with a mute button), dark mode, reduced-motion support, and a table view
5. **Static charts** (`outputs/*.png`): seasonal curves, the peak-date trend, cross-validation fits, and the 2026 forecast vs. actual.

## Run it
```bash
pip install -r requirements.txt
# data/raw/ already holds the 1979–2026 ADF&G counts and the yearly preseason table.
# To refresh them: export Kenai River (late-run sockeye) from the ADF&G portal to data/raw/fish_counts.csv
python run.py fetch-env   # telemetry -> data/raw/environment.csv + data/raw/preseason.csv
python run.py train
python run.py forecast    # 2026 preseason forecast
python run.py compare     # 2026 forecast vs actual, once 2026 counts are in the export
python run.py dashboard   # -> outputs/dashboard.html
python run.py site        # -> outputs/site/index.html (for hosting; fonts load from Google Fonts)
```
`python run.py synth` writes synthetic data for testing the pipeline without any downloads.

## Credits
Fonts: Caveat and Inter, both under the SIL Open Font License (see `dashboard/fonts/`).

"""Project settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

# Target fish counts
# Kenai River, Alaska: late-run sockeye counted by the ADF&G sonar at river mile 19.
# Sonar counts for late-run sockeye start July 1.
FISH_COUNTS_CSV = RAW / "fish_counts.csv"   # export from https://www.adfg.alaska.gov/sf/FishCounts/
SPECIES = "Sockeye"
SPECIES_ID = 420
COUNT_LOCATION_ID = 40    # Kenai River late run
RUN = "late"              # Kenai late run
LATE_RUN_START_DOY = 182  # July 1, when the late-run sonar count starts

# Environmental data
# USGS NWIS daily values. 00060 = discharge (cfs), 00010 = water temperature (C)
USGS_SITES = {
    "kenai_soldotna": "15266300",        # Kenai R at Soldotna (near the sonar)
    "kenai_cooper_landing": "15258000",  # Kenai R at Cooper Landing (upper river)
}
USGS_PARAMS = {"00060": "discharge_cfs", "00010": "water_temp_c"}

# Air temperature, precipitation, snowfall (snowmelt proxy) from Open-Meteo's archive
LAT, LON = 60.49, -151.06  # Soldotna, AK

# Sea-surface temperature from NOAA OISST via ERDDAP (Cook Inlet / Gulf of Alaska box)
SST_LAT, SST_LON = 59.5, -152.0  # lower Cook Inlet

START_YEAR, END_YEAR = 1979, 2025
PRESEASON_CSV = RAW / "preseason.csv"  # one row per year of conditions known before the run
FORECAST_YEAR = 2026
SEASON_DOY = (182, 240)  # July 1 to Aug 28 (sonar season)

# Credits shown in the dashboard footer
TEAM = [
    {"name": "Vivienne Le", "linkedin": "https://www.linkedin.com/in/le-vivienne",
     "role": "Designed this page in HTML and worked on the Python code"},
    {"name": "Earthan Le", "linkedin": "https://www.linkedin.com/in/earthan-le/",
     "role": "Worked on the research and the Python code"},
]

"""
features.py - shared data loading + feature engineering for SurfCast Sri Lanka.

Used by BOTH the Jupyter notebook and the Streamlit dashboard, so the model
always sees features built in exactly the same way.

AI-use declaration: parts of this code were generated with Claude (Anthropic)
and reviewed/run by the student.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("SURF_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw"))

# ---------------------------------------------------------------------------
# Surf spots. Directions are "coming from" angles in degrees (meteorological).
#   offshore_from : wind direction that blows from land to sea (clean waves)
#   swell_window  : swell directions that actually reach the break
# ---------------------------------------------------------------------------
SPOTS = {
    "Arugam Bay": {
        "key": "arugam", "lat": 6.84, "lon": 81.84,
        "coast": "East coast (best season roughly Apr-Oct, south-west monsoon)",
        "offshore_from": 270,          # westerly wind blows offshore on the east coast
        "swell_window": (110, 210),    # S-SE swells wrap into the point
    },
    "Hikkaduwa": {
        "key": "hikkaduwa", "lat": 6.14, "lon": 80.10,
        "coast": "South-west coast (best season roughly Nov-Apr, north-east monsoon)",
        "offshore_from": 70,           # north-easterly / easterly wind blows offshore
        "swell_window": (190, 290),    # SW-W swells
    },
}

# Surf rules used to define a "good surf day" (the label)
RULES = {
    "swell_min_m": 0.8,     # big enough to ride
    "swell_max_m": 2.5,     # not dangerous / closed-out for most surfers
    "period_min_s": 9.0,    # long-period (groundswell) waves are cleaner and stronger
    "light_wind_kmh": 20.0, # any direction is OK if the wind is this light
    "offshore_wind_kmh": 35.0,  # stronger wind is OK only if it is offshore
    "offshore_tolerance_deg": 60,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_open_meteo_csv(path: str) -> pd.DataFrame:
    """Open-Meteo CSV files start with a small metadata block, then a blank line,
    then the real table that begins with 'time,'. This finds the table."""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.lower().startswith("time,")), None)
    if start is None:
        raise ValueError(f"{os.path.basename(path)} does not look like an Open-Meteo CSV "
                         "(no header line starting with 'time,'). Did the link show an error?")
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[start:])))
    df.columns = [c.split(" (")[0].strip() for c in df.columns]   # drop units, e.g. 'wave_height_max (m)'
    df["time"] = pd.to_datetime(df["time"])
    return df


def _find(pattern_a: str, pattern_b: str) -> str:
    hits = [p for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
            if pattern_a in os.path.basename(p).lower() and pattern_b in os.path.basename(p).lower()]
    if not hits:
        raise FileNotFoundError(f"Could not find a '{pattern_a}*{pattern_b}*.csv' file in {DATA_DIR}. "
                                "Run download_data.py first.")
    return sorted(hits)[0]


def load_spot(spot: str) -> pd.DataFrame:
    key = SPOTS[spot]["key"]
    marine = _read_open_meteo_csv(_find(key, "marine"))
    weather = _read_open_meteo_csv(_find(key, "weather"))
    df = marine.merge(weather, on="time", how="inner")
    df.insert(0, "spot", spot)
    return df


def load_all() -> pd.DataFrame:
    """Both spots stacked, rows with missing core wave data dropped."""
    df = pd.concat([load_spot(s) for s in SPOTS], ignore_index=True)
    # Open-Meteo's default marine model only has wave/swell data from Oct 2021, and some
    # variables (e.g. swell peak period) can be completely empty: drop all-empty columns
    df = df.dropna(axis=1, how="all")
    core = ["swell_wave_height_max", "swell_wave_period_max", "swell_wave_direction_dominant",
            "wind_speed_10m_max", "wind_direction_10m_dominant"]
    df = df.dropna(subset=[c for c in core if c in df.columns]).reset_index(drop=True)
    return df.sort_values(["spot", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------
def angle_diff(a, b):
    """Smallest absolute difference between two compass angles (0..180)."""
    d = np.abs((np.asarray(a, dtype=float) - b + 180) % 360 - 180)
    return d


def good_surf(df: pd.DataFrame) -> pd.Series:
    """1 if the day is a 'good surf day' under RULES, else 0 (row-wise, per spot)."""
    out = np.zeros(len(df), dtype=int)
    for spot, cfg in SPOTS.items():
        m = (df["spot"] == spot).values
        d = df.loc[m]
        lo, hi = cfg["swell_window"]
        size_ok = d["swell_wave_height_max"].between(RULES["swell_min_m"], RULES["swell_max_m"])
        period_ok = d["swell_wave_period_max"] >= RULES["period_min_s"]
        dir_ok = d["swell_wave_direction_dominant"].between(lo, hi)
        offshore = angle_diff(d["wind_direction_10m_dominant"], cfg["offshore_from"]) <= RULES["offshore_tolerance_deg"]
        wind_ok = (d["wind_speed_10m_max"] <= RULES["light_wind_kmh"]) | (
            offshore & (d["wind_speed_10m_max"] <= RULES["offshore_wind_kmh"]))
        out[m] = (size_ok & period_ok & dir_ok & wind_ok).astype(int).values
    return pd.Series(out, index=df.index, name="good_surf")


# ---------------------------------------------------------------------------
# Features (only information known at the end of "today")
# ---------------------------------------------------------------------------
def sincos(values, period):
    rad = 2 * np.pi * np.asarray(values, dtype=float) / period
    return np.sin(rad), np.cos(rad)


RAW_NUMERIC = [  # used if present in the files
    "swell_wave_height_max", "swell_wave_period_max", "swell_wave_peak_period_max",
    "wave_height_max", "wave_period_max", "wind_wave_height_max", "wind_wave_period_max",
    "wind_speed_10m_max", "wind_gusts_10m_max", "precipitation_sum", "precipitation_hours",
    "temperature_2m_max", "relative_humidity_2m_mean", "pressure_msl_mean", "cloud_cover_mean",
]
DIRECTIONS = ["swell_wave_direction_dominant", "wave_direction_dominant",
              "wind_wave_direction_dominant", "wind_direction_10m_dominant"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level derived features (no history needed). Also used by the dashboard what-if sliders."""
    df = df.copy()
    for c in DIRECTIONS:
        if c in df.columns:
            df[c + "_sin"], df[c + "_cos"] = sincos(df[c], 360)
    off = df["spot"].map({s: c["offshore_from"] for s, c in SPOTS.items()})
    mid = df["spot"].map({s: np.mean(c["swell_window"]) for s, c in SPOTS.items()})
    # + means wind blowing offshore, - means onshore
    df["offshore_wind_component"] = df["wind_speed_10m_max"] * np.cos(np.deg2rad(df["wind_direction_10m_dominant"] - off))
    # 1 = swell coming straight into the break, -1 = from behind the coast
    df["swell_alignment"] = np.cos(np.deg2rad(df["swell_wave_direction_dominant"] - mid))
    df["wave_energy_proxy"] = df["swell_wave_height_max"] ** 2 * df["swell_wave_period_max"]
    df["month_sin"], df["month_cos"] = sincos(df["time"].dt.month, 12)
    df["doy_sin"], df["doy_cos"] = sincos(df["time"].dt.dayofyear, 365.25)
    df["is_arugam"] = (df["spot"] == "Arugam Bay").astype(int)
    return df


def build_dataset(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Returns (dataset, feature_columns). Target = good_surf_tomorrow."""
    if df is None:
        df = load_all()
    df = df.sort_values(["spot", "time"]).reset_index(drop=True)
    df["good_surf"] = good_surf(df)
    df = add_derived(df)

    g = df.groupby("spot", group_keys=False)
    df["swell_h_lag1"] = g["swell_wave_height_max"].shift(1)
    df["swell_h_lag2"] = g["swell_wave_height_max"].shift(2)
    df["swell_h_change"] = df["swell_wave_height_max"] - df["swell_h_lag1"]
    df["swell_p_lag1"] = g["swell_wave_period_max"].shift(1)
    df["wind_lag1"] = g["wind_speed_10m_max"].shift(1)
    if "pressure_msl_mean" in df.columns:
        df["pressure_change"] = df["pressure_msl_mean"] - g["pressure_msl_mean"].shift(1)
    df["swell_h_roll3"] = g["swell_wave_height_max"].transform(lambda s: s.rolling(3).mean())
    df["swell_h_roll7"] = g["swell_wave_height_max"].transform(lambda s: s.rolling(7).mean())
    df["wind_roll3"] = g["wind_speed_10m_max"].transform(lambda s: s.rolling(3).mean())
    df["good_surf_roll7"] = g["good_surf"].transform(lambda s: s.rolling(7).mean())

    # Targets: TOMORROW (shift -1 inside each spot) - must not leak into features
    df["good_surf_tomorrow"] = g["good_surf"].shift(-1)
    df["swell_h_tomorrow"] = g["swell_wave_height_max"].shift(-1)
    # guard against gaps in dates: tomorrow must really be the next day
    next_time = g["time"].shift(-1)
    df.loc[(next_time - df["time"]).dt.days != 1, ["good_surf_tomorrow", "swell_h_tomorrow"]] = np.nan

    feats = [c for c in RAW_NUMERIC if c in df.columns]
    feats += [c for c in df.columns if c.endswith("_sin") or c.endswith("_cos")]
    feats += [c for c in ["offshore_wind_component", "swell_alignment", "wave_energy_proxy", "is_arugam",
                          "good_surf", "good_surf_roll7", "swell_h_lag1", "swell_h_lag2", "swell_h_change",
                          "swell_p_lag1", "wind_lag1", "pressure_change", "swell_h_roll3", "swell_h_roll7",
                          "wind_roll3"] if c in df.columns]
    feats = list(dict.fromkeys(feats))
    df = df.dropna(subset=feats + ["good_surf_tomorrow", "swell_h_tomorrow"]).reset_index(drop=True)
    df["good_surf_tomorrow"] = df["good_surf_tomorrow"].astype(int)
    # global time order (needed for TimeSeriesSplit / blocked K-fold)
    df = df.sort_values(["time", "spot"]).reset_index(drop=True)
    return df, feats


def time_split(df: pd.DataFrame, test_years: int = 3):
    """Train on older years, test on the most recent `test_years` full years."""
    last_year = int(df["time"].dt.year.max())
    test_start = pd.Timestamp(year=last_year - test_years + 1, month=1, day=1)
    train = df[df["time"] < test_start]
    test = df[df["time"] >= test_start]
    return train, test, test_start


def apply_what_if(row: pd.Series, **changes) -> pd.Series:
    """Dashboard 'what-if': change today's raw values (e.g. swell height, wind speed) on one
    feature row and recompute every feature that depends on them."""
    r = row.copy()
    old_h = float(r["swell_wave_height_max"])
    for k, v in changes.items():
        r[k] = v
    one = pd.DataFrame([r])
    one["time"] = pd.to_datetime(one["time"])
    one = add_derived(one)
    one["good_surf"] = good_surf(one).values
    new_h = float(one["swell_wave_height_max"].iloc[0])
    if "swell_h_change" in one:
        one["swell_h_change"] = new_h - one["swell_h_lag1"]
    if "swell_h_roll3" in one:
        one["swell_h_roll3"] = one["swell_h_roll3"] + (new_h - old_h) / 3
    if "swell_h_roll7" in one:
        one["swell_h_roll7"] = one["swell_h_roll7"] + (new_h - old_h) / 7
    if "wind_roll3" in one and "wind_speed_10m_max" in changes:
        one["wind_roll3"] = one["wind_roll3"] + (changes["wind_speed_10m_max"] - float(row["wind_speed_10m_max"])) / 3
    if "good_surf_roll7" in one:
        one["good_surf_roll7"] = one["good_surf_roll7"] + (one["good_surf"] - int(row["good_surf"])) / 7
    return one.iloc[0]


FRIENDLY = {
    "swell_wave_height_max": "Swell height today (m)",
    "swell_wave_period_max": "Swell period today (s)",
    "swell_wave_peak_period_max": "Swell peak period today (s)",
    "wave_height_max": "Total wave height today (m)",
    "wind_speed_10m_max": "Max wind speed today (km/h)",
    "wind_gusts_10m_max": "Wind gusts today (km/h)",
    "offshore_wind_component": "Offshore wind component (km/h, + = offshore)",
    "swell_alignment": "Swell alignment with the break",
    "good_surf": "Good surf TODAY (persistence)",
    "good_surf_roll7": "Share of good days in last 7",
}

from __future__ import annotations
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import streamlit as st
import pydeck as pdk
import requests
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


# =========================================================
# 0) Utilities
# =========================================================
def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[Missing {what}] Cannot find: {path}")


def ensure_cols(df: pd.DataFrame, required: list[str], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{where}] Missing required columns: {missing}")


def clean_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def to_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def get_inspection_date(df: pd.DataFrame) -> pd.Series:
    if "inspection_date" in df.columns:
        return pd.to_datetime(df["inspection_date"], errors="coerce")
    if "INSPECTION DATE" in df.columns:
        return pd.to_datetime(df["INSPECTION DATE"], errors="coerce")
    raise ValueError("Need 'inspection_date' or 'INSPECTION DATE' column.")


def download_file(url: str, dst: Path, label: str = "") -> None:
    """Download url -> dst if dst missing/empty. Streams with progress bar."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        return

    with st.spinner(f"Downloading {label or dst.name} ..."):
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()

        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        progress = st.progress(0) if total > 0 else None

        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress.progress(min(downloaded / total, 1.0))

        tmp.replace(dst)
        if progress is not None:
            progress.empty()


def download_and_extract_zip(url: str, zip_dst: Path, extract_to: Path, label: str = "") -> None:
    """
    Download zip -> zip_dst, then extract into extract_to.
    Safe to call repeatedly (will skip extract if target shp exists).
    """
    extract_to.mkdir(parents=True, exist_ok=True)

    if any(extract_to.rglob("*.shp")):
        return

    download_file(url, zip_dst, label=label or zip_dst.name)

    with st.spinner(f"Extracting {label or zip_dst.name} ..."):
        with zipfile.ZipFile(zip_dst, "r") as zf:
            zf.extractall(extract_to)


def find_shapefile(root: Path, preferred_name: str = "nta_2020.shp") -> Path:
    """
    Find a .shp under root. Prefer exact file name if present.
    Works whether the zip contains a folder or just files.
    """
    exact = list(root.rglob(preferred_name))
    if exact:
        return exact[0]

    any_shp = list(root.rglob("*.shp"))
    if any_shp:
        return any_shp[0]

    raise FileNotFoundError(f"Cannot find any .shp under {root}")


# =========================================================
# 1) Paths
# =========================================================
THIS_FILE = Path(__file__).resolve()
REPO = THIS_FILE.parents[1]

DERIVED = REPO / "data" / "derived-data"
RAW_DIR = REPO / "data" / "raw-data"
NTA_DIR = RAW_DIR / "nta_2020"

SUMMARY_CSV = DERIVED / "nta_inspection_summary.csv"
INSP_PARQUET = DERIVED / "inspections_with_nta_income.parquet"
SUMMARY_CSV_YEAR = DERIVED / "nta_inspection_summary_by_year.csv"

# NOTE: RAW_NTA_SHP will be determined after ensuring data exists
RAW_NTA_SHP: Path | None = None


# =========================================================
# 1.5) Remote data URLs
# =========================================================
DATA_URLS = {
    "INSP_PARQUET": "https://github.com/junyu-hou/30538-final-project/releases/download/v1.0-data/inspections_with_nta_income.parquet",
    "SUMMARY_CSV": "https://github.com/junyu-hou/30538-final-project/releases/download/v1.0-data/nta_inspection_summary.csv",
    "NTA_ZIP": "https://github.com/junyu-hou/30538-final-project/releases/download/v1.0-data/nta_2020.zip",
    "SUMMARY_CSV_YEAR": "https://github.com/junyu-hou/30538-final-project/releases/download/v1.0-data/nta_inspection_summary_by_year.csv",
}


# =========================================================
# 2) Streamlit config
# =========================================================
INTENSITY_LABEL = "Inspection Intensity (Inspections per Restaurant per Year)"
INTENSITY_SHORT = "Inspections / restaurant / year"

st.set_page_config(page_title="NYC Restaurants — Income, Enforcement, Risk", layout="wide")
st.title("NYC Restaurant Inspections: Income, Enforcement Intensity, and Food Safety Risk")
st.caption("Left: NTA choropleth (PyDeck + CARTO basemap) | Right: scatter | Bottom: year slider")


# =========================================================
# 3) Ensure data exists (download if missing)
# =========================================================
try:
    # Derived data
    if not SUMMARY_CSV.exists():
        download_file(DATA_URLS["SUMMARY_CSV"], SUMMARY_CSV, "summary CSV")

    if not SUMMARY_CSV_YEAR.exists():
        download_file(DATA_URLS["SUMMARY_CSV_YEAR"], SUMMARY_CSV_YEAR, "summary CSV (by year)")

    if not INSP_PARQUET.exists():
        download_file(DATA_URLS["INSP_PARQUET"], INSP_PARQUET, "inspections parquet")

    if not any(NTA_DIR.rglob("*.shp")):
        zip_dst = RAW_DIR / "nta_2020.zip"
        download_and_extract_zip(DATA_URLS["NTA_ZIP"], zip_dst, NTA_DIR, "NTA shapefile zip")

    RAW_NTA_SHP = find_shapefile(NTA_DIR, preferred_name="nta_2020.shp")

except Exception as e:
    st.error(f"Data download/setup failed: {e}")
    st.stop()


# =========================================================
# 4) Cached loaders
# =========================================================
@st.cache_data(show_spinner=False)
def load_nta_geometry(shp_path: str) -> gpd.GeoDataFrame:
    shp = Path(shp_path)
    ensure_exists(shp, "NTA shapefile")

    gdf = gpd.read_file(shp).copy()

    candidates = ["nta2020", "NTA2020", "nta", "NTA"]
    nta_code_col = None
    for c in candidates:
        if c in gdf.columns:
            nta_code_col = c
            break
    if nta_code_col is None:
        raise ValueError(
            f"[NTA shapefile] Cannot find NTA code column among {candidates}. "
            f"Columns: {list(gdf.columns)[:30]}"
        )

    gdf["nta"] = clean_key(gdf[nta_code_col])
    gdf = gdf.to_crs(epsg=4326)
    return gdf[["nta", "geometry"]].copy()


@st.cache_data(show_spinner=False)
def load_summary(summary_csv: str, summary_csv_year: str) -> pd.DataFrame:
    """
    Load NTA-level summary (one row per NTA) for income/score,
    then merge in annualized inspection intensity computed from NTA-year file.
    """
    p = Path(summary_csv)
    ensure_exists(p, "inspection summary CSV")

    # ---- base summary (1 row per NTA): income + avg_score ----
    df = pd.read_csv(p).copy()
    ensure_cols(df, ["nta", "median_income_proxy", "avg_score"], "summary CSV")
    df["nta"] = clean_key(df["nta"])
    to_numeric_inplace(df, ["median_income_proxy", "avg_score"])

    df = df.dropna(subset=["median_income_proxy"]).copy()
    df = df[df["median_income_proxy"] > 0].copy()
    df["log_income"] = np.log(df["median_income_proxy"])

    # ---- annualized intensity from NTA-year summary ----
    py = Path(summary_csv_year)
    ensure_exists(py, "inspection summary CSV (by year)")

    dfy = pd.read_csv(py).copy()
    ensure_cols(dfy, ["nta", "year", "n_inspections", "n_unique_restaurants"], "summary CSV (by year)")
    dfy["nta"] = clean_key(dfy["nta"])
    dfy["year"] = pd.to_numeric(dfy["year"], errors="coerce")
    to_numeric_inplace(dfy, ["n_inspections", "n_unique_restaurants"])
    dfy = dfy.dropna(subset=["year"]).copy()

    # annual intensity per NTA-year
    dfy["intensity_year"] = dfy["n_inspections"] / dfy["n_unique_restaurants"]
    dfy.loc[dfy["n_unique_restaurants"] == 0, "intensity_year"] = np.nan
    dfy["intensity_year"] = dfy["intensity_year"].replace([np.inf, -np.inf], np.nan)

    # weighted average across years (weights = unique restaurants in that year)
    def _wavg(g: pd.DataFrame) -> float:
        x = g["intensity_year"].to_numpy(dtype=float)
        w = g["n_unique_restaurants"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if mask.sum() == 0:
            return np.nan
        return float(np.average(x[mask], weights=w[mask]))

    intensity_annual = (
        dfy.groupby("nta", as_index=False)
        .apply(lambda g: pd.Series({"inspection_intensity": _wavg(g)}))
        .reset_index(drop=True)
    )

    # merge annualized intensity into base df
    df = df.merge(intensity_annual, on="nta", how="left")

    return df


@st.cache_data(show_spinner=False)
def load_inspections(insp_parquet: str) -> pd.DataFrame:
    p = Path(insp_parquet)
    ensure_exists(p, "inspections parquet")

    df = pd.read_parquet(p).copy()
    ensure_cols(df, ["nta", "CAMIS", "median_income_proxy"], "inspections parquet")

    df["nta"] = clean_key(df["nta"])
    df["median_income_proxy"] = pd.to_numeric(df["median_income_proxy"], errors="coerce")
    df["inspection_date"] = get_inspection_date(df)

    df = df[df["inspection_date"].notna()].copy()
    df = df[df["inspection_date"] != pd.Timestamp("1900-01-01")].copy()
    df = df.dropna(subset=["nta", "CAMIS", "median_income_proxy"]).copy()
    df = df[df["median_income_proxy"] > 0].copy()

    df["year"] = df["inspection_date"].dt.year
    df["log_income"] = np.log(df["median_income_proxy"])
    return df


@st.cache_data(show_spinner=False)
def build_yearly_nta_panel(_insp_df: pd.DataFrame) -> pd.DataFrame:
    # NOTE: leading underscore avoids Streamlit hashing issues on some versions
    nta_year = (
        _insp_df.groupby(["year", "nta"], as_index=False)
        .agg(
            n_inspections=("CAMIS", "size"),
            n_unique_restaurants=("CAMIS", "nunique"),
            log_income=("log_income", "first"),
        )
        .copy()
    )
    nta_year["inspection_intensity"] = nta_year["n_inspections"] / nta_year["n_unique_restaurants"]
    nta_year = nta_year.replace([np.inf, -np.inf], np.nan)
    nta_year = nta_year.dropna(subset=["log_income", "inspection_intensity"]).copy()
    return nta_year


# =========================================================
# 5) Transforms
# =========================================================
def make_map_gdf(nta_gdf: gpd.GeoDataFrame, summary_df: pd.DataFrame) -> gpd.GeoDataFrame:
    keep = ["nta", "median_income_proxy", "log_income", "avg_score", "inspection_intensity"]
    return nta_gdf.merge(summary_df[keep], on="nta", how="left")


# =========================================================
# 6) PyDeck choropleth (CARTO basemap, no token) + tooltip
# =========================================================
def choropleth_pydeck(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    legend_title: str,
    cmap_name: str = "viridis",
    height: int = 520,
):
    d = gdf[["nta", value_col, "geometry"]].copy()
    d = d[d["geometry"].notna()].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    # formatted string for tooltip
    if value_col == "median_income_proxy":
        d["value_str"] = d[value_col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "Missing")
    else:
        d["value_str"] = d[value_col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "Missing")

    # bounds -> center view
    minx, miny, maxx, maxy = d.total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2

    # normalize for colormap
    v = d[value_col].to_numpy()
    if np.isfinite(v).any():
        vmin = float(np.nanmin(v))
        vmax = float(np.nanmax(v))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax == vmin:
        vmax = vmin + 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap_name)

    def to_rgba(x):
        if pd.isna(x):
            return [220, 220, 220, 160]
        r, g, b, a = cmap(norm(float(x)))
        return [int(255 * r), int(255 * g), int(255 * b), 160]

    d["fill_color"] = d[value_col].apply(to_rgba)

    # GeoJSON FeatureCollection
    geojson = json.loads(d.to_json())

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[40, 40, 40],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=9.3)

    tooltip = {
        "html": (
            "<b>NTA:</b> {nta}<br/>"
            f"<b>{legend_title}:</b> " + "{value_str}"
        ),
        "style": {"backgroundColor": "white", "color": "black"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        height=height,
    )

    return deck, vmin, vmax


def render_colorbar_vertical(vmin: float, vmax: float, cmap_name: str, label: str):
    fig, ax = plt.subplots(figsize=(1.1, 4.2))
    fig.subplots_adjust(left=0.35, right=0.85, top=0.95, bottom=0.08)

    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="vertical")
    cb.set_label(label)
    st.pyplot(fig, clear_figure=True)


# =========================================================
# 7) Altair scatter
# =========================================================
def scatter_altair(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str | None = None,
    y_label: str | None = None,
    x_domain: tuple[float, float] | None = None,
    width: int = 520,
    height: int = 520,
) -> alt.Chart:
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y]).copy()

    x_enc = alt.X(f"{x}:Q", title=(x_label or x))
    if x_domain is not None:
        x_enc = x_enc.scale(domain=list(x_domain))

    base = (
        alt.Chart(d)
        .mark_circle(opacity=0.65)
        .encode(
            x=x_enc,
            y=alt.Y(f"{y}:Q", title=(y_label or y)),
            tooltip=[
                alt.Tooltip("nta:N", title="NTA") if "nta" in d.columns else alt.Tooltip(f"{x}:Q"),
                alt.Tooltip(f"{x}:Q", title=(x_label or x)),
                alt.Tooltip(f"{y}:Q", title=(y_label or y)),
            ],
        )
    )

    reg = base.transform_regression(x, y).mark_line(color="red")

    props = {"width": width, "height": height}
    if title:
        props["title"] = title

    return (base + reg).properties(**props)


# =========================================================
# 8) Load data
# =========================================================
with st.spinner("Loading data..."):
    assert RAW_NTA_SHP is not None
    nta_gdf = load_nta_geometry(str(RAW_NTA_SHP))
    summary_df = load_summary(str(SUMMARY_CSV), str(SUMMARY_CSV_YEAR))
    insp_df = load_inspections(str(INSP_PARQUET))
    nta_year = build_yearly_nta_panel(insp_df)

map_gdf = make_map_gdf(nta_gdf, summary_df)


# =========================================================
# 9) Layout
# =========================================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("NTA Map")

    map_metric = st.selectbox(
        "Select metric to display",
        options=[
            "Income (Median Household Income)",
            "Food Risk (Average Inspection Score)",
            "Enforcement (Inspection Intensity)",
        ],
        index=0,
    )

    if map_metric == "Income (Median Household Income)":
        value_col = "median_income_proxy"
        title = "NYC NTA Median Household Income"
        legend = "Median household income ($)"
        cmap_name = "viridis"

    elif map_metric == "Food Risk (Average Inspection Score)":
        value_col = "avg_score"
        title = "NYC NTA Food Risk (Average Inspection Score)"
        legend = "Avg score (higher = worse)"
        cmap_name = "Reds"

    else:
        value_col = "inspection_intensity"
        title = f"NYC NTA {INTENSITY_LABEL}"
        legend = INTENSITY_SHORT
        cmap_name = "cividis"

    st.markdown(f"**{title}**")

    map_col, leg_col = st.columns([5, 1], gap="small")

    deck, vmin, vmax = choropleth_pydeck(
        map_gdf,
        value_col=value_col,
        legend_title=legend,
        cmap_name=cmap_name,
        height=520,
    )

    with map_col:
        st.pydeck_chart(deck, use_container_width=True)

    with leg_col:
        render_colorbar_vertical(vmin, vmax, cmap_name, legend)

with col_right:
    st.subheader("Scatter (NTA-level)")

    VAR_LABELS = {
        "log_income": "Log Median Household Income",
        "inspection_intensity": INTENSITY_LABEL,
        "avg_score": "Average Inspection Score (Higher = Worse)",
    }

    x = st.selectbox(
        "X variable",
        options=list(VAR_LABELS.keys()),
        format_func=lambda k: VAR_LABELS[k],
        index=0,
    )
    y = st.selectbox(
        "Y variable",
        options=list(VAR_LABELS.keys()),
        format_func=lambda k: VAR_LABELS[k],
        index=1,
    )

    x_domain = (10.0, 13.0) if x == "log_income" else None
    chart_title = f"{VAR_LABELS[y]} vs {VAR_LABELS[x]} (NTA)"

    st.altair_chart(
        scatter_altair(
            map_gdf,
            x=x,
            y=y,
            title=chart_title,
            x_label=VAR_LABELS[x],
            y_label=VAR_LABELS[y],
            x_domain=x_domain,
        ),
        use_container_width=True,
    )


# =========================================================
# 10) Year slider
# =========================================================
st.divider()
st.subheader(f"Over time: Income vs {INTENSITY_LABEL} (by year)")

years = sorted([int(y) for y in nta_year["year"].dropna().unique()])

if years:
    year = st.slider(
        "Select year",
        min_value=min(years),
        max_value=max(years),
        value=max(years),
        step=1,
        key="year_slider_time",
    )

    sub = nta_year[nta_year["year"] == year].copy()
    st.caption(f"Year = {year} | NTA rows = {len(sub):,}")

    st.markdown(
        f"**{INTENSITY_LABEL} vs Log Median Household Income (NTA)**  \n"
        f"Year: **{year}**"
    )

    st.altair_chart(
        scatter_altair(
            sub.assign(nta=sub["nta"]),
            x="log_income",
            y="inspection_intensity",
            title="",
            x_label="Log Median Household Income",
            y_label=INTENSITY_LABEL,
            x_domain=(10.0, 13.0),
            width=1100,
            height=380,
        ),
        use_container_width=True,
    )
else:
    st.warning("No valid years found (check inspection_date cleaning).")
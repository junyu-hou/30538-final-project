# streamlit_app.py (PyDeck with CARTO basemap, no Mapbox token needed)
# Save to: final_project/code/streamlit_app.py
# Run (from repo root): streamlit run code/streamlit_app.py

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import streamlit as st
import pydeck as pdk

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


# =========================================================
# 1) Paths  (STRICTLY RELATIVE)
# =========================================================
if Path("data").exists() and Path("code").exists():
    REPO = Path(".")
else:
    REPO = Path("..")

RAW_NTA_SHP = REPO / "data" / "raw-data" / "nta_2020" / "nta_2020.shp"
DERIVED = REPO / "data" / "derived-data"

SUMMARY_CSV = DERIVED / "nta_inspection_summary.csv"
INSP_PARQUET = DERIVED / "inspections_with_nta_income.parquet"


# =========================================================
# 2) Streamlit config
# =========================================================
st.set_page_config(page_title="NYC Restaurants — Income, Enforcement, Risk", layout="wide")
st.title("NYC Restaurant Inspections: Income, Enforcement Intensity, and Food Safety Risk")
st.caption("Left: NTA choropleth (PyDeck + CARTO basemap) | Right: scatter | Bottom: year slider")


# =========================================================
# 3) Cached loaders
# =========================================================
@st.cache_data(show_spinner=False)
def load_nta_geometry(shp_path: str) -> gpd.GeoDataFrame:
    shp = Path(shp_path)
    ensure_exists(shp, "NTA shapefile")

    gdf = gpd.read_file(shp).copy()
    nta_code_col = "nta2020"
    ensure_cols(gdf, [nta_code_col], "NTA shapefile")

    gdf["nta"] = clean_key(gdf[nta_code_col])
    gdf = gdf.to_crs(epsg=4326)
    return gdf[["nta", "geometry"]].copy()


@st.cache_data(show_spinner=False)
def load_summary(summary_csv: str) -> pd.DataFrame:
    p = Path(summary_csv)
    ensure_exists(p, "inspection summary CSV")

    df = pd.read_csv(p).copy()
    ensure_cols(
        df,
        ["nta", "median_income_proxy", "n_inspections", "n_unique_restaurants", "avg_score"],
        "summary CSV",
    )
    df["nta"] = clean_key(df["nta"])
    to_numeric_inplace(df, ["median_income_proxy", "n_inspections", "n_unique_restaurants", "avg_score"])

    df["inspection_intensity"] = df["n_inspections"] / df["n_unique_restaurants"]
    df.loc[df["n_unique_restaurants"] == 0, "inspection_intensity"] = np.nan
    df["inspection_intensity"] = df["inspection_intensity"].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=["median_income_proxy"]).copy()
    df = df[df["median_income_proxy"] > 0].copy()
    df["log_income"] = np.log(df["median_income_proxy"])
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
def build_yearly_nta_panel(insp_df: pd.DataFrame) -> pd.DataFrame:
    nta_year = (
        insp_df.groupby(["year", "nta"], as_index=False)
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
# 4) Transforms
# =========================================================
def make_map_gdf(nta_gdf: gpd.GeoDataFrame, summary_df: pd.DataFrame) -> gpd.GeoDataFrame:
    keep = ["nta", "median_income_proxy", "log_income", "avg_score", "inspection_intensity"]
    return nta_gdf.merge(summary_df[keep], on="nta", how="left")


# =========================================================
# 5) PyDeck choropleth (CARTO basemap, no token) + tooltip
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

    # IMPORTANT: For many pydeck builds, tooltip variables work as {field} (not {properties.field})
    # Since GeoJSON properties are available as top-level keys in tooltip templating, use {nta} and {value_str}.
    tooltip = {
        "html": (
            "<b>NTA:</b> {nta}<br/>"
            f"<b>{legend_title}:</b> " + "{value_str}"
        ),
        "style": {"backgroundColor": "white", "color": "black"},
    }

    # CARTO basemap (no Mapbox token needed)
    # If your pydeck build doesn’t accept these, it will gracefully fall back to no basemap.
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
# 6) Altair scatter
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
# 7) Load data
# =========================================================
with st.spinner("Loading data..."):
    nta_gdf = load_nta_geometry(str(RAW_NTA_SHP))
    summary_df = load_summary(str(SUMMARY_CSV))
    insp_df = load_inspections(str(INSP_PARQUET))
    nta_year = build_yearly_nta_panel(insp_df)

map_gdf = make_map_gdf(nta_gdf, summary_df)


# =========================================================
# 8) Layout
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
        title = "NYC NTA Inspection Intensity (Inspections per Restaurant)"
        legend = "Inspections / restaurant"
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
        "inspection_intensity": "Inspection Intensity (Inspections per Restaurant)",
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

    x_domain = (10.0, 13) if x == "log_income" else None

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
# 9) Year slider
# =========================================================
st.divider()
st.subheader("Over time: Income vs Enforcement Intensity (by year)")

VAR_LABELS = {
    "log_income": "Log Median Household Income",
    "inspection_intensity": "Inspection Intensity (Inspections per Restaurant)",
}

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
        f"**{VAR_LABELS['inspection_intensity']} vs {VAR_LABELS['log_income']} (NTA)**  \n"
        f"Year: **{year}**"
    )

    x_domain = (10.0, 13.0)

    st.altair_chart(
        scatter_altair(
            sub.assign(nta=sub["nta"]),
            x="log_income",
            y="inspection_intensity",
            title="",
            x_label=VAR_LABELS["log_income"],
            y_label=VAR_LABELS["inspection_intensity"],
            x_domain=x_domain,
            width=1100,
            height=380,
        ),
        use_container_width=True,
    )

else:
    st.warning("No valid years found (check inspection_date cleaning).")
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

RAW = Path("data/raw-data")
DERIVED = Path("data/derived-data")
DERIVED.mkdir(parents=True, exist_ok=True)


def read_acs_households(path: Path) -> pd.DataFrame:
    """
    ACS B11001: Total households (Estimate).
    Expect columns like: GEO_ID, NAME, B11001_001E ...
    """
    df = pd.read_csv(path, dtype=str)
    # keep header row that has actual data (ACS files already ok)
    df["geoid"] = df["GEO_ID"].str.extract(r"US(\d+)$")[0]
    df["households"] = pd.to_numeric(df["B11001_001E"], errors="coerce")
    return df[["geoid", "households"]]


def read_acs_income(path: Path) -> pd.DataFrame:
    """
    ACS B19013: Median household income (Estimate).
    Expect columns like: GEO_ID, NAME, B19013_001E ...
    """
    df = pd.read_csv(path, dtype=str)
    df["geoid"] = df["GEO_ID"].str.extract(r"US(\d+)$")[0]
    df["median_income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
    return df[["geoid", "median_income"]]


def load_tracts(path: Path) -> gpd.GeoDataFrame:
    """
    TIGER/Line tracts. We only need GEOID and geometry.
    """
    gdf = gpd.read_file(path)
    # TIGER has GEOID field; ensure string
    gdf["geoid"] = gdf["GEOID"].astype(str)
    return gdf[["geoid", "geometry"]].copy()


def load_ntas(path: Path) -> gpd.GeoDataFrame:
    """
    NYC NTA boundaries shapefile. Field names vary; we try common ones.
    """
    gdf = gpd.read_file(path)
    # Common NTA code fields: 'NTACode', 'NTA2020', 'nta2020', 'NTA'
    candidates = ["NTACode", "NTA2020", "nta2020", "NTA", "ntacode"]
    nta_col = next((c for c in candidates if c in gdf.columns), None)
    if nta_col is None:
        raise ValueError(f"Could not find NTA code field in columns: {list(gdf.columns)}")

    gdf = gdf.rename(columns={nta_col: "nta"})
    gdf["nta"] = gdf["nta"].astype(str)
    return gdf[["nta", "geometry"]].copy()


def attach_acs_to_tracts(tracts: gpd.GeoDataFrame, acs: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Attribute join ACS (tract-level) onto tract polygons by geoid.
    """

    tract_candidates = ["geoid", "GEOID", "GEO_ID"]
    tract_geoid = next((c for c in tract_candidates if c in tracts.columns), None)

    if tract_geoid is None:
        raise ValueError(f"Could not find GEOID in tract columns: {tracts.columns}")

    tracts = tracts.rename(columns={tract_geoid: "geoid"})
    tracts["geoid"] = tracts["geoid"].astype(str)

    acs["geoid"] = acs["geoid"].astype(str)

    out = tracts.merge(acs, on="geoid", how="left")

    return out


def compute_nta_income_households_from_tracts(
    tracts_with_acs: gpd.GeoDataFrame,
    ntas: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Build tract→NTA overlay, allocate households by area share,
    then compute NTA households and household-weighted median income proxy.

    Note: Median is not strictly aggregatable. A common approximation is household-weighted mean of tract medians.
    (You should mention this limitation in writeup.)
    """
    # Use an equal-area projection for accurate area/intersection
    # EPSG:2263 (NAD83 / New York Long Island) is common for NYC
    target_crs = "EPSG:2263"
    tr = tracts_with_acs.to_crs(target_crs).copy()
    nt = ntas.to_crs(target_crs).copy()

    # Clean missing
    tr["households"] = pd.to_numeric(tr["households"], errors="coerce")
    tr["median_income"] = pd.to_numeric(tr["median_income"], errors="coerce")
    tr = tr[(tr["households"] > 0) & (tr["median_income"].notna())].copy()

    # tract area
    tr["tract_area"] = tr.geometry.area

    # intersection overlay 
    inter = gpd.overlay(
        tr[["geoid", "households", "median_income", "tract_area", "geometry"]],
        nt[["nta", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    inter["inter_area"] = inter.geometry.area
    inter["area_share"] = inter["inter_area"] / inter["tract_area"]

    # Allocate households to each piece by area share
    inter["hh_alloc"] = inter["households"] * inter["area_share"]

    # compute weighted sum safely
    inter["income_hh"] = inter["median_income"] * inter["hh_alloc"]
    tmp = inter.groupby("nta", as_index=False).agg(
        households=("hh_alloc", "sum"),
        income_hh=("income_hh", "sum"),
    )
    tmp["median_income_proxy"] = tmp["income_hh"] / tmp["households"]

    out = tmp[["nta", "households", "median_income_proxy"]].copy()
    return out


def load_inspections_points(path: Path) -> gpd.GeoDataFrame:
    """
    Load inspections CSV and create point geometry.
    Uses Latitude/Longitude columns in your screenshot.
    """
    df = pd.read_csv(path)

    # Parse dates, handle 1900-01-01 placeholder
    if "INSPECTION DATE" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["INSPECTION DATE"], errors="coerce")
        df = df[df["inspection_date"].notna()].copy()
        df = df[df["inspection_date"] != pd.Timestamp("1900-01-01")].copy()

    # Ensure numeric lat/lon
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )
    return gdf


def assign_nta_to_inspections(inspections: gpd.GeoDataFrame, ntas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatial join points → NTA.
    """
    # project both to same CRS for sjoin
    nt = ntas.to_crs(inspections.crs)
    joined = gpd.sjoin(inspections, nt[["nta", "geometry"]], how="left", predicate="within")
    # geopandas adds index_right
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    return joined


def main():
    # ---- Paths (yours) ----
    inspections_path = RAW / "inspections_nyc.csv"

    nta_shp = RAW / "nta_2020" / "nta_2020.shp"  
    tract_shp = RAW / "census_tract_2022" / "tl_2022_36_tract.shp"

    acs_hh_path = RAW / "acs_total_households_2024" / "ACSDT5Y2024.B11001-Data.csv"
    acs_inc_path = RAW / "acs_median_income_2024" / "ACSDT5Y2024.B19013-Data.csv"

    # ---- 1) ACS merge (tract level) ----
    hh = read_acs_households(acs_hh_path)
    inc = read_acs_income(acs_inc_path)
    acs = hh.merge(inc, on="geoid", how="inner")
    acs = acs.dropna(how="all")

    acs_out = DERIVED / "acs_tract_2024.csv"
    acs.to_csv(acs_out, index=False)

    # ---- 2) Attach ACS to tract polygons ----
    tracts = load_tracts(tract_shp)
    tracts_acs = attach_acs_to_tracts(tracts, acs)

    tract_gpkg = DERIVED / "tract_acs.gpkg"
    tracts_acs.to_file(tract_gpkg, layer="tract_acs", driver="GPKG")

    # ---- 3) Tract → NTA aggregation (income + households) ----
    ntas = load_ntas(nta_shp)
    nta_demo = compute_nta_income_households_from_tracts(tracts_acs, ntas)

    nta_demo_out = DERIVED / "nta_income_households_2024.csv"
    nta_demo.to_csv(nta_demo_out, index=False)

    # ---- 4) Assign NTA to inspections (keep point-level) ----
    insp = load_inspections_points(inspections_path)

    # keep original NTA column if exists, but also create nta_from_geom
    insp_joined = assign_nta_to_inspections(insp, ntas)
    insp_joined = insp_joined.rename(columns={"nta": "nta_from_geom"})

    # If your CSV already has 'NTA' code column, keep it too:
    # (In your screenshot there's a column named 'NTA' near the end.)
    # We'll also create a final nta column preferring geometry-based.
    if "NTA" in insp_joined.columns:
        insp_joined["nta_from_file"] = insp_joined["NTA"].astype(str)
    else:
        insp_joined["nta_from_file"] = pd.NA

    insp_joined["nta"] = insp_joined["nta_from_geom"].fillna(insp_joined["nta_from_file"])

    # ---- 5) Merge NTA demographics back onto each inspection ----
    insp_final = insp_joined.merge(nta_demo, on="nta", how="left")

    # Save point-level
    insp_out = DERIVED / "inspections_with_nta_income.parquet"
    insp_final.to_parquet(insp_out, index=False)

    # Optional: NTA-level inspection summary
    # (you can add year/month later)
    if "SCORE" in insp_final.columns:
        insp_final["SCORE"] = pd.to_numeric(insp_final["SCORE"], errors="coerce")
    if "GRADE" in insp_final.columns:
        insp_final["is_A"] = (insp_final["GRADE"].astype(str).str.upper() == "A").astype(float)

    summary = (
        insp_final.groupby("nta", as_index=False)
        .agg(
            n_inspections=("CAMIS", "size"),
            n_unique_restaurants=("CAMIS", "nunique"),
            avg_score=("SCORE", "mean"),
            share_grade_A=("is_A", "mean"),
            households=("households", "first"),
            median_income_proxy=("median_income_proxy", "first"),
        )
    )
    summary_out = DERIVED / "nta_inspection_summary.csv"
    summary.to_csv(summary_out, index=False)

    print("Saved outputs:")
    print(" -", acs_out)
    print(" -", tract_gpkg)
    print(" -", nta_demo_out)
    print(" -", insp_out)
    print(" -", summary_out)


if __name__ == "__main__":
    main()
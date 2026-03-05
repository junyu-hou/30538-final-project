# Income, Enforcement Intensity, and Restaurant Inspection Outcomes in New York City

## 1. Project Overview
This project studies how Neighborhood Tabulation Area (NTA) income relates to restaurant inspection enforcement across New York City. We combine spatial mapping, regression-based scatter plots, time-trend analysis, and a streamlit dashboard to evaluate whether inspection intensity and food-safety outcomes are equitably distributed across NTAs.

## 2. Data Sources
### 1. DOHMH New York City Restaurant Inspection Results

- Source: NYC OpenData
- Link: (https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data)
- Description: Provides restaurant inspection results and violation records.

### 2. American Community Survey (ACS) 2024 5-Year Estimates

- Source: United States Census Bureau
- Link: (https://data.census.gov/table/ACSDT1Y2024.B19013?q=B19013), (https://data.census.gov/table?q=B11001)
- Description: Provides median household income and total household at the census tract level.

### 3. TIGER/Line Shapefiles (2022)

- Source: United States Census Bureau
- Link: (https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2022.html#list-tab-790442341)
- Description: Provides census tract boundary for NYC.

### 4. 2020 Neighborhood Tabulation Areas (NTAs) - Mapped

- Source: NYC OpenData
- Link: (https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-Mapped/4hft-v355)
- Description: Provides NYC Neighborhood Tabulation Area (NTA) boundaries.


Since our data exceed GitHub size limits:

- Download from: (https://drive.google.com/drive/folders/1hLaZLTzCRmQGFMZt1qWDVw8VTslaARs5?usp=share_link)
- Save to: final_project/data


Reproducibility Instructions:

1. Clone the repository
2. Create environment and install dependencies
3. Run the pipeline: 
- python code/preprocessing.py
- quarto render code/static_plots.qmd
- streamlit run streamlit-app/streamlit_app.py


## 3. Data Processing Pipeline
All data cleaning and integration are implemented in: code/preprocessing.py

---

### 1. ACS Processing (Tract Level)
We first merge two ACS tables by tract GEOID:
- B11001: Total Households  
- B19013: Median Household Income  
These attributes are joined to census tract polygons.

---

### 2. Tract → NTA Aggregation
To construct NTA-level demographic measures, we overlay Census tracts with NTAs in an equal-area coordinate reference system (EPSG:2263).
For each tract–NTA intersection piece, we compute an area share:

- The share of tract area that lies within each NTA
We then allocate tract households to NTAs proportionally based on this area share.
Total NTA households are computed as the sum of allocated tract households.
Finally, we construct an NTA-level income proxy using a household-weighted average of tract median household income.

---

### 3. Inspection Data Cleaning
Inspection records are cleaned in three main steps:

1. Parse Inspection Date and remove missing or placeholder dates (e.g., 1900-01-01).
2. Convert Latitude and Longitude to numeric format and drop invalid coordinates.
3. Convert inspections to spatial points (EPSG:4326) and spatially join them to NTA polygons.

---

### 4. Derived Outputs
Processed datasets will be saved to: data/derived-data/

Outputs include:

- A point-level integrated dataset (inspections_with_nta_income.parquet)
- An NTA-level summary file
- An NTA-by-year summary file for time-trend analysis


## 4. Static Plots
Static plots are generated using: code/static_plots.qmd

They include 3 maps, s scatter pots, 1 stcaked bar chart, and 1 line chart:

1. NYC Income Map (Median Household Income by NTA)
2. NYC Average Inspection Score Map (Average Score Map by NTA)
3. NYC Inspection Intensity Map (Inspections pre Restaurants per Year) by NTA
4. Income and Inspection Intensity Across NYC NTAs
5. Income and Food Safety Risk Measured by Average Score (NTA)
6. Inspection Intensity and Food Safety Risk Measured by Average Score (NTA)
7. Inspection Stage Shares by Income Quintile
8. Yearly Slope of Inspection Intensity on Income (NTA-Level)


## 5. Streamlit Dashboard
Git Release is used when deploy the Streamlit

The interactive dashboard allows users to:

- Switch between three NTA-level choropleth maps.
- Choose the X and Y variables to explore their relationships. A fitted trend line is shown to summarize the overall association between the selected variables.
- Examine how the income–intensity relationship changes over time.

Streamlit dashboard link: [(https://30538-final-project-phshfr3n9aqnepymn6buyq.streamlit.app)]


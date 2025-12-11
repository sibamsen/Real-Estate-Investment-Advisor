# src/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from io import StringIO, BytesIO
import zipfile

# helper functions from your utilities
from eda_utils import is_good_investment, forecast_price_fixed, forecast_price_by_location

# --- Configuration / Load data ---
zip_path = Path(__file__).parent.parent / "cleaned_india_housing_prices.zip"  
csv_name = "cleaned_india_housing_prices.csv"

if zip_path.exists():
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(z.open(csv_name))
else:
    # fallback to direct CSV if zip not present
    csv_path = Path(__file__).parent.parent / "cleaned_india_housing_prices.csv"
    df = pd.read_csv(csv_path))

st.set_page_config(page_title="Real Estate Investment Advisor (No-ML)", layout='wide')
col_logo, col_title = st.columns([1, 14])
with col_logo:
    # show logo if exists, otherwise ignore
    logo_path = Path(__file__).parent / "logo.jpg"
    if logo_path.exists():
        st.image(str(logo_path), width=60)
with col_title:
    st.title("Real Estate Investment Advisor")
st.markdown("**Based on the project brief (Data Preprocessing + EDA + Streamlit).**")

# ---------- Precompute & safe conversions ----------
# map Yes/No to numeric for common columns (if present)
yes_no_map = {'yes': 1, 'y': 1, 'true': 1, '1': 1, 'no': 0, 'n': 0, 'false': 0, '0': 0}
for col in ['parking_space', 'nearby_schools', 'nearby_hospitals']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().map(yes_no_map)
# amenity count (clean)
if 'amenities' in df.columns:
    df['amenities_count'] = df['amenities'].fillna('').apply(
        lambda x: len([a for a in [s.strip() for s in x.split(',')] if a != ''])
    )
else:
    df['amenities_count'] = 0

# ensure numeric fields exist
for ncol in ['price_in_lakhs', 'size_in_sqft', 'price_per_sqft', 'age_of_property']:
    if ncol not in df.columns:
        df[ncol] = np.nan
# drop rows with invalid sizes to avoid division errors (if needed)
df = df[df['size_in_sqft'].replace(0, np.nan).notna()]

# ---------- Sidebar: Filters ----------
st.sidebar.header("Filter properties")
state = st.sidebar.selectbox("State", options=['All'] + sorted(df['state'].dropna().unique().tolist()))
city = st.sidebar.selectbox("City", options=['All'] + sorted(df['city'].dropna().unique().tolist()))
min_price = int(np.nanmin(df['price_in_lakhs'])) if df['price_in_lakhs'].notna().any() else 0
max_price = int(np.nanmax(df['price_in_lakhs'])) if df['price_in_lakhs'].notna().any() else 1
price_range = st.sidebar.slider("Price (Lakhs)", min_value=min_price, max_value=max_price,
                                value=(min_price, max_price))
bhk_options = sorted(df['bhk'].dropna().unique().tolist()) if 'bhk' in df.columns else []
bhk = st.sidebar.multiselect("BHK", options=bhk_options, default=[])

# Checkbox to optionally save EDA figures to outputs/figures
save_figs = st.sidebar.checkbox("Save EDA figures to outputs/figures", value=False)

# ---------- Apply Filters ----------
filtered = df.copy()
if state != 'All':
    filtered = filtered[filtered['state'] == state]
if city != 'All':
    filtered = filtered[filtered['city'] == city]
filtered = filtered[(filtered['price_in_lakhs'] >= price_range[0]) & (filtered['price_in_lakhs'] <= price_range[1])]
if bhk:
    filtered = filtered[filtered['bhk'].isin(bhk)]

st.markdown(f"### Filtered properties: **{len(filtered)}** found")

# ---------- Top row: KPI cards ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total listings (filtered)", f"{len(filtered)}")
with col2:
    median_price = filtered['price_in_lakhs'].median()
    st.metric("Median price (Lakhs)", f"{median_price:.2f}" if not np.isnan(median_price) else "N/A")
with col3:
    avg_pps = filtered['price_per_sqft'].mean()
    st.metric("Avg price per sqft (₹)", f"{avg_pps:.0f}" if not np.isnan(avg_pps) else "N/A")
with col4:
    # top city by median price
    top_city = df.groupby('city')['price_in_lakhs'].median().sort_values(ascending=False).index[0] if df['city'].notna().any() else "N/A"
    st.metric("Top city (median price)", f"{top_city}")

# ---------- Table + selection ----------
st.dataframe(filtered[['id', 'state', 'city', 'locality', 'property_type', 'bhk', 'size_in_sqft', 'price_in_lakhs']].head(200))

st.markdown("### Select a property by ID to see investment analysis")
prop_id = st.selectbox("Property ID", options=['None'] + filtered['id'].astype(str).tolist())

if prop_id != 'None':
    prop = filtered[filtered['id'].astype(str) == prop_id].iloc[0].to_dict()
    st.subheader("Property details")
    st.json({k: prop.get(k) for k in ['state', 'city', 'locality', 'property_type', 'bhk', 'size_in_sqft', 'price_in_lakhs', 'price_per_sqft'] if k in prop})

    # Investment decision (rule-based)
    decision, score = is_good_investment(prop, df)
    st.metric(label="Investment decision", value=decision, delta=f"score={score}")

    # Forecast options
    st.markdown("#### Forecast price after 5 years")
    method = st.radio("Choose forecast method", options=['Fixed Rate (8%)', 'City-based rate', 'Custom rate'])
    current_price_rupees = prop['price_in_lakhs'] * 100000

    if method == 'Fixed Rate (8%)':
        est = forecast_price_fixed(current_price_rupees, r=0.08, t=5)
    elif method == 'City-based rate':
        city_name = prop['city']
        city_med = df[df['city'] == city_name]['price_in_lakhs'].median()
        national_med = df['price_in_lakhs'].median()
        growth = 0.06 if (not np.isnan(city_med) and city_med > national_med) else 0.04
        est = forecast_price_by_location(current_price_rupees, growth, t=5)
    else:
        r = st.number_input("Annual growth rate (e.g., 0.08 for 8%)", value=0.08, step=0.01)
        est = forecast_price_fixed(current_price_rupees, r=r, t=5)

    st.write(f"Estimated future price after 5 years: ₹{est:,.0f} (≈ {est/100000:.2f} Lakhs)")

# ---------- EDA Section ----------
st.markdown("## EDA Visuals")

# 1) Price distribution (histogram)
fig_price = px.histogram(filtered, x='price_in_lakhs', nbins=50, title='Price Distribution (Lakhs)', marginal='box')
st.plotly_chart(fig_price, width='stretch')
if save_figs:
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig_price.write_image("outputs/figures/price_distribution.png", engine="kaleido")

# 2) Size distribution
fig_size = px.histogram(filtered, x='size_in_sqft', nbins=50, title='Size Distribution (sqft)', marginal='box')
st.plotly_chart(fig_size, width='stretch')
if save_figs:
    fig_size.write_image("outputs/figures/size_distribution.png", engine="kaleido")

# 3) Scatter size vs price (interactive)
fig_scatter = px.scatter(filtered, x='size_in_sqft', y='price_in_lakhs', color='city',
                         hover_data=['locality', 'property_type'], title='Size vs Price (Lakhs)')
st.plotly_chart(fig_scatter, width='stretch')
if save_figs:
    fig_scatter.write_image("outputs/figures/size_vs_price.png", engine="kaleido")

# 4) Correlation heatmap (numeric features)
num_cols = ['price_in_lakhs', 'size_in_sqft', 'price_per_sqft', 'age_of_property', 'nearby_schools', 'nearby_hospitals', 'parking_space', 'amenities_count']
# keep only existing columns
num_cols = [c for c in num_cols if c in df.columns]
num_df = df[num_cols].apply(pd.to_numeric, errors='coerce')
# fill (for display only) with median to avoid missing blocking the heatmap
num_df = num_df.fillna(num_df.median())
corr = num_df.corr()
fig_corr = px.imshow(corr, text_auto='.2f', title='Correlation matrix (numeric features)')
st.plotly_chart(fig_corr, width='stretch')
if save_figs:
    fig_corr.write_image("outputs/figures/correlation_matrix.png", engine="kaleido")

# 5) Amenities analysis: amenities_count vs price_per_sqft with trendline
if 'amenities_count' in df.columns and 'price_per_sqft' in df.columns:
    fig_amen = px.scatter(filtered, x='amenities_count', y='price_per_sqft', trendline='ols',
                         title='Amenities count vs Price per sqft', hover_data=['locality', 'city'])
    st.plotly_chart(fig_amen, width='stretch')
    if save_figs:
        fig_amen.write_image("outputs/figures/amenities_vs_price.png", engine="kaleido")

    # show top 10 most common amenities (bar)
    if 'amenities' in df.columns:
        # explode amenities
        all_amen = df['amenities'].fillna('').apply(lambda x: [a.strip() for a in x.split(',') if a.strip() != ''])
        flat = [item for sublist in all_amen for item in sublist]
        top_amen = pd.Series(flat).value_counts().head(10).reset_index()
        top_amen.columns = ['amenity', 'count']
        fig_top_amen = px.bar(top_amen, x='amenity', y='count', title='Top 10 Amenities', color='amenity')
        st.plotly_chart(fig_top_amen, width='stretch')
        if save_figs:
            fig_top_amen.write_image("outputs/figures/top_10_amenities.png", engine="kaleido")

# 6) Price per sqft by top localities (original chart)
top_localities = df.groupby('locality')['price_per_sqft'].median().sort_values(ascending=False).head(10).reset_index()
fig_local = px.bar(top_localities, x='locality', y='price_per_sqft', title='Top 10 localities by median price per sqft', color='locality')
st.plotly_chart(fig_local, width='stretch')
if save_figs:
    fig_local.write_image("outputs/figures/top_localities_pps.png", engine="kaleido")

# ---------- Download cleaned dataset and filtered dataset ----------
with st.expander("Download datasets"):
    colA, colB = st.columns(2)
    with colA:
        # Read CSV bytes from the zip and provide them to the download button
        with zipfile.ZipFile(zip_path) as z:
            cleaned_bytes = z.read("cleaned_india_housing_prices.csv")
        st.download_button("Download cleaned dataset (CSV)", data=cleaned_bytes, file_name="cleaned_india_housing_prices.csv", mime="text/csv")
    with colB:
        # download currently filtered dataframe
        csv_bytes = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download filtered results (CSV)", data=csv_bytes, file_name="filtered_india_housing_prices.csv", mime="text/csv")

# ---------- Methodology / Assumptions ----------
st.markdown("## Methodology & Assumptions")
st.markdown("""
**Preprocessing**
- Standardized column names, removed duplicates, converted numeric columns.
- Created `price_per_sqft` = (price_in_lakhs * 100000) / size_in_sqft.
- Computed `age_of_property` from `year_built` where available.
- Converted Yes/No features (parking_space, nearby_schools, nearby_hospitals) to 1/0.

**Investment decision (rule-based, no ML)**
- Score points for: price <= city median, price_per_sqft <= city median, bhk >= 3, availability = 'available'.
- If score >= 3 → 'Good Investment' else 'Not Good Investment'.

**Forecast**
- Options: Fixed rate (default 8% p.a.), City-based proxy (0.06/0.04 based on city vs national median), or custom rate input.
""")

# End of app


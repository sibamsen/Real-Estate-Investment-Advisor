# src/eda_utils.py
"""
Helper utilities for EDA + rule-based investment decision + simple forecasts.

Functions:
- normalize_yes_no_columns(df, cols)
- compute_amenities_count(df, amenities_col='amenities')
- get_top_amenities(df, n=10, amenities_col='amenities')
- is_good_investment(property_row, df, threshold=3)
- forecast_price_fixed(current_price_in_rupees, r=0.08, t=5)
- forecast_price_by_location(current_price_in_rupees, location_growth_rate, t=5)
- city_growth_proxy(df, city, high_rate=0.06, low_rate=0.04)
- prepare_numeric_df_for_corr(df, extra_bool_cols=None)
"""

from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import math

_yes_no_map = {
    'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1,
    'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0
}


def normalize_yes_no_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Map common Yes/No-like string values to 1/0 for columns in `cols`.
    Operates in-place on a copy and returns the updated DataFrame.
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(_yes_no_map)
            )
    return df


def compute_amenities_count(df: pd.DataFrame, amenities_col: str = 'amenities') -> pd.DataFrame:
    """
    Create/overwrite `amenities_count` column by counting comma-separated items in amenities_col.
    Returns modified copy of df.
    """
    df = df.copy()
    if amenities_col in df.columns:
        df['amenities_count'] = df[amenities_col].fillna('').apply(
            lambda x: len([a for a in [s.strip() for s in x.split(',')] if a != ''])
        )
    else:
        df['amenities_count'] = 0
    return df


def get_top_amenities(df: pd.DataFrame, n: int = 10, amenities_col: str = 'amenities') -> pd.DataFrame:
    """
    Return a DataFrame with top `n` amenities and their counts.
    Columns: ['amenity', 'count']
    """
    if amenities_col not in df.columns:
        return pd.DataFrame(columns=['amenity', 'count'])

    all_lists = df[amenities_col].fillna('').apply(
        lambda x: [a for a in [s.strip() for s in x.split(',')] if a != '']
    )

    flat = [item for lst in all_lists for item in lst]
    if len(flat) == 0:
        return pd.DataFrame(columns=['amenity', 'count'])

    top = pd.Series(flat).value_counts().head(n).reset_index()
    top.columns = ['amenity', 'count']
    return top


def is_good_investment(property_row: Union[pd.Series, dict], df: pd.DataFrame, threshold: int = 3) -> Tuple[str, int]:
    """
    Rule-based decision if a property is a 'Good Investment' or not.

    Rules (each gives +1 point):
      - price_in_lakhs <= city median price_in_lakhs
      - price_per_sqft <= city median price_per_sqft
      - bhk >= 3
      - availability_status == 'available' (case-insensitive check)

    property_row can be a Pandas Series or a dict (e.g. the selected row).
    Returns tuple: (decision_str, score)
    """
    # allow dict or Series
    if isinstance(property_row, pd.Series):
        row = property_row.to_dict()
    else:
        row = dict(property_row)

    score = 0

    # safe getters with fallback
    city = row.get('city', None)
    try:
        price = float(row.get('price_in_lakhs', np.nan))
    except Exception:
        price = np.nan
    try:
        pps = float(row.get('price_per_sqft', np.nan))
    except Exception:
        pps = np.nan
    try:
        bhk = int(row.get('bhk')) if row.get('bhk') not in (None, '', np.nan) else 0
    except Exception:
        # some bhk values like '2 RK' may break int conversion -> fallback 0
        try:
            bhk = int(float(str(row.get('bhk')).split()[0]))
        except Exception:
            bhk = 0
    availability = str(row.get('availability_status', '')).strip().lower()

    # city medians (fallback to global median if city empty or med NaN)
    if city and 'city' in df.columns:
        city_df = df[df['city'] == city]
        city_med_price = city_df['price_in_lakhs'].median() if not city_df.empty else np.nan
        city_med_pps = city_df['price_per_sqft'].median() if not city_df.empty else np.nan
    else:
        city_med_price = np.nan
        city_med_pps = np.nan

    global_med_price = df['price_in_lakhs'].median() if 'price_in_lakhs' in df.columns else np.nan
    global_med_pps = df['price_per_sqft'].median() if 'price_per_sqft' in df.columns else np.nan

    # Rule 1
    if not math.isnan(price):
        cmp_med = city_med_price if (not math.isnan(city_med_price) and not np.isclose(city_med_price, 0)) else global_med_price
        if not math.isnan(cmp_med) and price <= cmp_med:
            score += 1

    # Rule 2
    if not math.isnan(pps):
        cmp_pps_med = city_med_pps if (not math.isnan(city_med_pps) and not np.isclose(city_med_pps, 0)) else global_med_pps
        if not math.isnan(cmp_pps_med) and pps <= cmp_pps_med:
            score += 1

    # Rule 3
    if bhk >= 3:
        score += 1

    # Rule 4 (availability)
    if availability and 'avail' in availability:  # 'available', 'availability: available' etc.
        score += 1
    else:
        # also consider some common synonyms
        if availability in ('ready-to-move', 'ready to move', 'ready', 'available'):
            score += 1

    decision = 'Good Investment' if score >= threshold else 'Not Good Investment'
    return decision, score


def forecast_price_fixed(current_price_in_rupees: float, r: float = 0.08, t: int = 5) -> float:
    """
    Compound growth forecast: price * (1+r)^t
    """
    try:
        return float(current_price_in_rupees) * ((1.0 + float(r)) ** int(t))
    except Exception:
        return float('nan')


def forecast_price_by_location(current_price_in_rupees: float, location_growth_rate: float, t: int = 5) -> float:
    """
    Same as fixed but using a location-specific growth rate.
    """
    return forecast_price_fixed(current_price_in_rupees, r=location_growth_rate, t=t)


def city_growth_proxy(df: pd.DataFrame, city: str, high_rate: float = 0.06, low_rate: float = 0.04) -> float:
    """
    Return a simple proxy annual growth rate for a given city.
    Heuristic used:
      - If dataset contains multiple years and a 'year' column that can be used to compute
        a simple average year-over-year growth, try that.
      - Otherwise, compare city median price to national median and return high_rate if city median > national median, else low_rate.
    """
    # Attempt time-series growth if dataset has year-like columns and prices across years
    possible_year_cols = [c for c in df.columns if 'year' in c.lower()]
    if possible_year_cols:
        # Try to find a year column and compute crude CAGR if multiple years present
        for yc in possible_year_cols:
            try:
                temp = df[[yc, 'price_in_lakhs']].dropna()
                temp[yc] = pd.to_numeric(temp[yc], errors='coerce')
                if temp[yc].nunique() >= 2:
                    # aggregate median price by year for the city (if possible)
                    city_mask = (df['city'] == city) if ('city' in df.columns) and (city is not None) else pd.Series([True] * len(df))
                    yearly = df[city_mask].dropna(subset=[yc, 'price_in_lakhs']).groupby(yc)['price_in_lakhs'].median().sort_index()
                    if len(yearly) >= 2:
                        # compute simple CAGR between first and last year
                        y0 = yearly.iloc[0]
                        yN = yearly.iloc[-1]
                        n = yearly.index[-1] - yearly.index[0]
                        if n > 0 and y0 > 0:
                            cagr = (yN / y0) ** (1.0 / n) - 1.0
                            # clip to reasonable range
                            if not math.isnan(cagr) and math.isfinite(cagr):
                                return float(np.clip(cagr, -0.5, 0.5))  # keep in [-50%, 50%]
            except Exception:
                continue

    # Fallback heuristic: compare medians
    if 'price_in_lakhs' not in df.columns:
        return low_rate
    city_med = df[df['city'] == city]['price_in_lakhs'].median() if city is not None and 'city' in df.columns else np.nan
    global_med = df['price_in_lakhs'].median()
    try:
        if not math.isnan(city_med) and city_med > global_med:
            return float(high_rate)
    except Exception:
        pass
    return float(low_rate)


def prepare_numeric_df_for_corr(df: pd.DataFrame, extra_bool_cols: List[str] = None) -> pd.DataFrame:
    """
    Return a numeric-only DataFrame suitable for correlation calculations and heatmaps.

    - Maps common yes/no columns to 0/1.
    - Adds 'amenities_count' if missing.
    - Keeps numeric columns only and returns a copy with NaNs filled by median (so corr() won't fail).
    """
    df_num = df.copy()
    base_bool_cols = ['parking_space', 'nearby_schools', 'nearby_hospitals']
    if extra_bool_cols:
        base_bool_cols += extra_bool_cols

    # Normalize boolean-style columns
    df_num = normalize_yes_no_columns(df_num, base_bool_cols)

    # amenities count
    df_num = compute_amenities_count(df_num)

    # Select numeric-like columns
    numeric = df_num.select_dtypes(include=[np.number]).copy()

    # If some important numeric columns are strings, try coercion
    for col in ['price_in_lakhs', 'size_in_sqft', 'price_per_sqft', 'age_of_property', 'amenities_count']:
        if col in df_num.columns and col not in numeric.columns:
            numeric[col] = pd.to_numeric(df_num[col], errors='coerce')

    # Fill NaNs with median to allow correlation plotting
    numeric = numeric.fillna(numeric.median())

    return numeric

# End of file

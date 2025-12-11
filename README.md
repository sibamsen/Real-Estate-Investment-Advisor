# Real Estate Investment Advisor (Data Preprocessing + EDA + Streamlit Dashboard)

This project is a complete **Real Estate Analytics Dashboard** built using **Python, Pandas, Plotly, and Streamlit**.  
It transforms raw housing data through **data preprocessing**, performs **exploratory data analysis (EDA)**, and presents insights via an interactive dashboard.  
It also provides a **rule-based investment decision system** and a **5-year price forecast** (No Machine Learning).

---

## 🚀 Live Dashboard  
🔗 **Streamlit App:** [https://your-streamlit-link.streamlit.app/](https://real-estate-investment-advisor-webapp.streamlit.app/)

---

## 🧹 1. Data Preprocessing

Performed in `preprocessing.ipynb` and applied in the Streamlit app.

### ✔ Key Steps
- Handling missing values  
- Standardizing column names  
- Converting Yes/No → 0/1  
- Creating new features:
  - `price_per_sqft`
  - `amenities_count`
  - `age_of_property`
- Removing invalid/zero values
- IQR-based outlier treatment  
- Exporting final cleaned dataset  

---

## 📊 2. Exploratory Data Analysis (EDA)

The dashboard includes:

### ✔ Distributions
- Price distribution  
- Size distribution  

### ✔ Relationship Analysis
- Size vs Price scatter  
- Amenities count vs Price per sqft  
- Price per sqft vs locality (Top 10 localities)

### ✔ Correlation Analysis
- Full correlation heatmap on numeric features  

### ✔ Amenities Insights
- Most common amenities  
- Impact of amenities count on pricing  

### ✔ Location Insights
- City-wise median prices  
- Locality ranking by price per sqft  

---

## 🧠 3. Investment Decision (Rule-Based, No ML)

A property is labeled **Good Investment** if score ≥ 3:

| Rule | Condition |
|------|-----------|
| 1 | Price ≤ City median price |
| 2 | Price_per_sqft ≤ City median pps |
| 3 | BHK ≥ 3 |
| 4 | Availability = "Available" |

Displayed using `st.metric()`.

---

## 📈 4. Price Forecasting (5-Year Projection)

Three forecasting methods:

1. **Fixed Growth Rate (8%)**  
2. **City-Based Growth Rate**  
   - City median > National median → 6%  
   - Else → 4%  
3. **Custom Growth Rate** (user-defined)

No ML is used — purely mathematical forecasting.

---

## 🖥 5. Streamlit Dashboard Features

### ✔ Sidebar Filters
- State  
- City  
- BHK  
- Price range  

### ✔ KPI Cards
- Total listings  
- Median price  
- Average price per sqft  
- Top city by pricing  

### ✔ Property-Level Analysis
- Property details  
- Rule-based investment score  
- Forecasted price after 5 years  

### ✔ Interactive EDA Visuals
- Histograms  
- Scatter plots  
- Bar charts  
- Correlation heatmap  

### ✔ Downloads
- Cleaned dataset  
- Filtered dataset  

---

## 📝 Author  
**Sibam Sen**   
Data Analytics • Python • EDA • Streamlit

---

If you like this project, consider giving it a ⭐ on GitHub!

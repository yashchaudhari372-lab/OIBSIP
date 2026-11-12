import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Sales Prediction App | Oasis Infobyte", page_icon="📈", layout="wide")

st.title("📈 Advertising Sales Prediction Web App")
st.markdown("""
**Oasis Infobyte — SIP Data Science Internship (Task 5)**  
* **Intern:** Yash Devidas Chaudhari | **Reference:** OIB/N2/IP5780
""")
st.divider()

@st.cache_resource
def load_and_train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Advertising.csv")
    if not os.path.exists(csv_path):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv", csv_path)
    
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
        
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr.fit(X, y)
    
    return lr, gbr, df

lr, gbr, df = load_and_train()

st.sidebar.header("💰 Advertising Budget Allocation ($ in Thousands)")

model_choice = st.sidebar.selectbox("Predictive Algorithm", ["Gradient Boosting Regressor (Recommended - 98% R²)", "Linear Regression (Interpretable Baseline)"])
chosen_model = gbr if "Gradient Boosting" in model_choice else lr

tv_spend = st.sidebar.slider("📺 TV Advertising Spend ($k)", min_value=0.0, max_value=350.0, value=150.0, step=5.0)
radio_spend = st.sidebar.slider("📻 Radio Advertising Spend ($k)", min_value=0.0, max_value=60.0, value=25.0, step=1.0)
news_spend = st.sidebar.slider("📰 Newspaper Advertising Spend ($k)", min_value=0.0, max_value=120.0, value=20.0, step=2.0)

total_budget = tv_spend + radio_spend + news_spend

# Metrics Display
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Ad Budget", f"${total_budget:,.1f}k")
c2.metric("TV Share", f"{(tv_spend/total_budget*100) if total_budget > 0 else 0:.1f}%")
c3.metric("Radio Share", f"{(radio_spend/total_budget*100) if total_budget > 0 else 0:.1f}%")
c4.metric("Newspaper Share", f"{(news_spend/total_budget*100) if total_budget > 0 else 0:.1f}%")

st.markdown("### 🎯 Predicted Sales Output")
input_df = pd.DataFrame([[tv_spend, radio_spend, news_spend]], columns=['TV', 'Radio', 'Newspaper'])
pred_sales = chosen_model.predict(input_df)[0]
estimated_units = int(pred_sales * 1000)

st.success(f"### 📦 Estimated Product Sales: **{pred_sales:.2f} thousand units** (~**{estimated_units:,} units**)")

# Budget breakdown chart
st.subheader("📊 Advertising Investment Breakdown")
chart_df = pd.DataFrame({
    "Channel": ["TV", "Radio", "Newspaper"],
    "Budget ($k)": [tv_spend, radio_spend, news_spend]
}).set_index("Channel")
st.bar_chart(chart_df)

with st.expander("💡 Executive Business Insights"):
    st.markdown("""
    * **TV Advertising ($r = 0.78$):** The primary volume driver. Significant allocations here reliably scale overall demand.
    * **Radio Advertising ($r = 0.58$):** High marginal efficiency per dollar spent. Adding Radio alongside TV yields synergistic returns.
    * **Newspaper Advertising ($r = 0.23$):** Shows virtually zero independent impact once TV and Radio are accounted for. Reallocating Newspaper budget to TV/Radio maximizes marketing ROI.
    """)

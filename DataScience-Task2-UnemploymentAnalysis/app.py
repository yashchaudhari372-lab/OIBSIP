import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Unemployment Analysis | Oasis Infobyte", page_icon="📊", layout="wide")

st.title("📊 Unemployment in India Analysis Dashboard")
st.markdown("""
**Oasis Infobyte — SIP Data Science Internship (Task 2)**  
* **Intern:** Yash Devidas Chaudhari | **Reference:** OIB/N2/IP5780
""")
st.divider()

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Unemployment_in_India.csv")
    if not os.path.exists(csv_path):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/amankharwal/Website-data/master/unemployment.csv", csv_path)
    
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        'Region': 'State',
        'Region.1': 'Zone',
        'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
        'Estimated Employed': 'Employed',
        'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
    }, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%m-%Y', errors='coerce')
    df.dropna(subset=['Date', 'Unemployment_Rate'], inplace=True)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")
all_states = sorted(df['State'].unique().tolist())
selected_states = st.sidebar.multiselect("Select States for Comparison", all_states, default=['Maharashtra', 'Delhi', 'Tamil Nadu', 'Uttar Pradesh'])

all_zones = sorted(df['Zone'].unique().tolist())
selected_zone = st.sidebar.selectbox("Filter by Zone", ["All Zones"] + all_zones)

filtered_df = df.copy()
if selected_zone != "All Zones":
    filtered_df = filtered_df[filtered_df['Zone'] == selected_zone]

# Key Metric Cards
m1, m2, m3, m4 = st.columns(4)
overall_avg = filtered_df['Unemployment_Rate'].mean()
lockdown_date = pd.to_datetime('2020-03-24')
pre_mean = filtered_df[filtered_df['Date'] < lockdown_date]['Unemployment_Rate'].mean()
post_mean = filtered_df[filtered_df['Date'] >= lockdown_date]['Unemployment_Rate'].mean()
max_state = filtered_df.groupby('State')['Unemployment_Rate'].mean().idxmax()
max_rate = filtered_df.groupby('State')['Unemployment_Rate'].mean().max()

m1.metric("Avg Unemployment", f"{overall_avg:.2f}%")
m2.metric("Pre-COVID Mean", f"{pre_mean:.2f}%")
m3.metric("Post-COVID Mean", f"{post_mean:.2f}%", delta=f"+{post_mean - pre_mean:.2f}%")
m4.metric("Highest Impact State", f"{max_state} ({max_rate:.1f}%)")

st.markdown("### 📈 Unemployment Rate Over Time")
if selected_states:
    plot_df = df[df['State'].isin(selected_states)].sort_values('Date')
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.lineplot(data=plot_df, x='Date', y='Unemployment_Rate', hue='State', marker='o', ax=ax)
    ax.axvline(lockdown_date, color='red', linestyle='--', label='Lockdown Start')
    ax.set_title("Unemployment Trend Across Selected States", fontsize=12, fontweight='bold')
    ax.set_ylabel("Unemployment Rate (%)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("### 🏆 Top 10 Highest Unemployment States")
top10 = df.groupby('State')['Unemployment_Rate'].mean().sort_values(ascending=False).head(10)
chart_df = pd.DataFrame({'State': top10.index, 'Avg Unemployment Rate (%)': top10.values}).set_index('State')
st.bar_chart(chart_df)

with st.expander("ℹ️ Key Analytical Findings"):
    st.markdown("""
    * **COVID-19 Shock:** Nationwide unemployment jumped significantly during April–May 2020 following the immediate enforcement of the nationwide lockdown.
    * **Regional Vulnerabilities:** Northern states like Haryana, Tripura, and Jharkhand registered the highest average unemployment peaks exceeding 25-30%.
    * **Recovery Trends:** Urban and industrialized regions experienced rapid recovery by September–October 2020 as economic restrictions gradually relaxed.
    """)

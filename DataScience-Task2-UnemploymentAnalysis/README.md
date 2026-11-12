# 📊 Task 2: Unemployment Analysis with Python

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oibsip-unemployment-analysis-yash.streamlit.app/)  
🌐 **Live Web Application:** [https://oibsip-unemployment-analysis-yash.streamlit.app/](https://oibsip-unemployment-analysis-yash.streamlit.app/)

**Oasis Infobyte — SIP (Student Internship Program)**  
* **Intern Name:** Yash Devidas Chaudhari  
* **Track:** Data Science  
* **Offer Letter ID:** OIB/N2/IP5780  
* **Repository:** `OIBSIP`  
* **Folder Name:** `DataScience-Task2-UnemploymentAnalysis`  
* **Live Link:** https://oibsip-unemployment-analysis-yash.streamlit.app/  

---

## 📌 Project Overview
Unemployment is one of the most critical socio-economic barometers. The outbreak of COVID-19 in early 2020 and subsequent nationwide lockdown measures inflicted severe disruption on the Indian labor market. This project delivers a thorough Exploratory Data Analysis (EDA) of unemployment rates across Indian states, examining regional disparities, seasonal movements, and the sharp shock caused by the pandemic.

---

## 📋 Feature Checklist Compliance

| Requirement from Task Card | Status | Implementation Details |
| :--- | :---: | :--- |
| **Download Dataset** | ✅ | Bundled locally: `Unemployment_in_India.csv` covering all Indian states and regions. |
| **Data Loading & Cleaning** | ✅ | Stripped whitespace from column names, handled null values, parsed dates to datetime. |
| **Region & Month Trends** | ✅ | Computed region-wise averages and temporal month-by-month changes. |
| **Time-Series Line Chart** | ✅ | Visualized trajectory across major states (Maharashtra, Delhi, Tamil Nadu, UP, West Bengal). |
| **Top 10 States Bar Chart** | ✅ | Ranked Top 10 states with highest average unemployment (Haryana, Tripura, Jharkhand, etc.). |
| **Correlation Heatmap** | ✅ | Examined relationships between Unemployment Rate, Employed Population, and Labour Participation Rate. |
| **Pre vs. Post COVID Comparison** | ✅ | Quantified the impact before vs after March 2020 national lockdown (jump from 9.5% to 13.7%+). |
| **Written Observations** | ✅ | Accompanying markdown narratives and analytical conclusions for each visualization. |
| **Interactive Streamlit App** | ✅ | Built interactive dashboard with state multiselect, zone filters, and timeline comparisons. |

---

## 📊 Key Analytical Findings
* **The Lockdown Shock:** National unemployment surged dramatically following the March 24, 2020 lockdown, peaking during April–May 2020.
* **Top Impacted States:** Haryana, Tripura, and Jharkhand registered average unemployment rates between 25% and 27%.
* **Urban vs. Rural Disparity:** Industrialized urban states (Delhi, Tamil Nadu, Maharashtra) witnessed sharper acute spikes due to retail and service sector shutdowns.
* **Recovery Trajectory:** By September–October 2020, rates showed a distinct recovery trend toward pre-pandemic baselines as lockdown measures eased.

---

## 🛠️ Project Structure

```text
OIBSIP/
└── DataScience-Task2-UnemploymentAnalysis/
    ├── README.md
    ├── Unemployment_in_India.csv
    ├── unemployment_analysis.ipynb
    ├── train_and_predict.py
    ├── app.py
    └── images/
        ├── top10_states_unemployment.png
        ├── timeseries_unemployment.png
        ├── correlation_heatmap.png
        ├── precovid_vs_postcovid.png
        └── zone_distribution.png
```

---

## 🚀 How to Run Locally

### 1. Terminal Run (Instant Summary & Stats)
```bash
python train_and_predict.py
```

### 2. Interactive Jupyter Notebook
```bash
jupyter notebook unemployment_analysis.ipynb
```

### 3. Launch Interactive Web App
```bash
streamlit run app.py
```

---

## 📹 Video Walkthrough & Title Card
* **Title Card (First 2 Seconds):**
  * **Name:** Yash Devidas Chaudhari
  * **Track:** Data Science
  * **Task:** Task 2: Unemployment Analysis with Python
* **Walkthrough:** Explain the COVID-19 lockdown impact, demonstrate the time-series line chart, show the top 10 impacted states, and showcase the Streamlit dashboard!

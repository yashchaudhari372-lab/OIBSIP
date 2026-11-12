# 📈 Task 5: Sales Prediction Using Python

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oibsip-sales-prediction-yash.streamlit.app/)  
🌐 **Live Web Application:** [https://oibsip-sales-prediction-yash.streamlit.app/](https://oibsip-sales-prediction-yash.streamlit.app/)

**Oasis Infobyte — SIP (Student Internship Program)**  
* **Intern Name:** Yash Devidas Chaudhari  
* **Track:** Data Science  
* **Offer Letter ID:** OIB/N2/IP5780  
* **Repository:** `OIBSIP`  
* **Folder Name:** `DataScience-Task5-SalesPrediction`  
* **Live Link:** https://oibsip-sales-prediction-yash.streamlit.app/  

---

## 📌 Project Overview
Forecasting sales revenue from advertising investments across diverse channels (TV, Radio, Newspaper) is fundamental for strategic marketing and budget allocation. The objective of this project is to build and evaluate regression models that predict product sales based on advertising spend and reveal which medium delivers the greatest business impact.

---

## 📋 Feature Checklist Compliance

| Requirement from Task Card | Status | Implementation Details |
| :--- | :---: | :--- |
| **Download Dataset** | ✅ | Bundled locally: `Advertising.csv` (200 marketing records). |
| **Data Loading & EDA** | ✅ | Verified shape (200, 4), confirmed zero null values, generated summary statistics. |
| **Individual Scatter Plots** | ✅ | Plotted Sales vs. TV, Radio, and Newspaper with regression trendlines. |
| **Correlation Heatmap** | ✅ | Computed Pearson correlation matrix ($r_{TV}=0.78$, $r_{Radio}=0.58$, $r_{News}=0.23$). |
| **Train/Test Split** | ✅ | 80/20 train/test split (`random_state=42`). |
| **Baseline Linear Regression** | ✅ | Fitted baseline `LinearRegression()` with full coefficient interpretation. |
| **Additional Models** | ✅ | Trained **Ridge Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**. |
| **Evaluation Metrics** | ✅ | Evaluated Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score. |
| **Residual Diagnostic Plot** | ✅ | Checked error normality and confirmed residuals are randomly distributed around zero. |
| **Channel Interpretation** | ✅ | Concluded that TV and Radio drive virtually all sales, while Newspaper spend has near-zero effect. |
| **Interactive Streamlit App** | ✅ | Deployed `app.py` enabling real-time budget adjustments and sales forecasting. |

---

## 📊 Model Performance Comparison

| Model | $R^2$ Score | RMSE | MAE |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting Regressor** *(Best Model)* | **98.24%** | **0.672** | **0.573** |
| **Random Forest Regressor** | **98.13%** | **0.692** | **0.590** |
| **Linear Regression (Baseline)** | 89.94% | 1.782 | 1.461 |
| **Ridge Regression** | 89.93% | 1.783 | 1.462 |

---

## 💡 Executive Insights & Interpretation
* **TV Advertising ($r = 0.78$):** The primary volume driver accounting for **~60% of model feature importance**.
* **Radio Advertising ($r = 0.58$):** High marginal efficiency ($+0.189$ sales per \$1,000 spend).
* **Newspaper Advertising ($r = 0.23$):** Negligible impact ($+0.0028$ coefficient). Reallocating Newspaper budget into TV/Radio reliably boosts overall return on investment (ROI).

---

## 🛠️ Project Structure

```text
OIBSIP/
└── DataScience-Task5-SalesPrediction/
    ├── README.md
    ├── Advertising.csv
    ├── sales_prediction.ipynb
    ├── train_and_predict.py
    ├── app.py
    └── images/
        ├── pairplot.png
        ├── scatter_plots.png
        ├── correlation_heatmap.png
        ├── residuals_plot.png
        ├── model_comparison.png
        └── feature_importance.png
```

---

## 🚀 How to Run Locally

### 1. Terminal Run (Instant Prediction & Analysis)
```bash
python train_and_predict.py
```

### 2. Interactive Jupyter Notebook
```bash
jupyter notebook sales_prediction.ipynb
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
  * **Task:** Task 5: Sales Prediction Using Python
* **Walkthrough:** Show the dataset correlation, individual scatter plots with regression trendlines, model comparison bar chart, residual diagnostics, and the interactive Streamlit budget calculator!

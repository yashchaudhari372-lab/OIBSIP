# 🚗 Task 3: Car Price Prediction with Machine Learning

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oibsip-car-price-prediction-yash.streamlit.app/)  
🌐 **Live Web Application:** [https://oibsip-car-price-prediction-yash.streamlit.app/](https://oibsip-car-price-prediction-yash.streamlit.app/)

**Oasis Infobyte — SIP (Student Internship Program)**  
* **Intern Name:** Yash Devidas Chaudhari  
* **Track:** Data Science  
* **Offer Letter ID:** OIB/N2/IP5780  
* **Repository:** `OIBSIP`  
* **Folder Name:** `DataScience-Task3-CarPricePrediction`  
* **Live Link:** https://oibsip-car-price-prediction-yash.streamlit.app/  

---

## 📌 Project Overview
Automobile price estimation is influenced by a combination of brand perception, technical powertrain metrics, structural dimensions, and fuel economy. The objective of this project is to develop and evaluate Machine Learning regression models to predict automobile market valuations with high accuracy.

---

## 📋 Feature Checklist Compliance

| Requirement from Task Card | Status | Implementation Details |
| :--- | :---: | :--- |
| **Download Dataset** | ✅ | Bundled locally: `CarPrice.csv` (205 automotive records with 26 technical specs). |
| **Data Cleaning** | ✅ | Handled missing values, verified duplicates, and resolved brand typographical discrepancies (`vokswagen`, `toyouta`, `porcshce`). |
| **Feature Engineering** | ✅ | Extracted clean `Brand` feature from car names (22 unique manufacturers). |
| **Comprehensive EDA** | ✅ | Visualized selling price distribution, price by body/fuel type, and scatter plots for horsepower/engine capacity. |
| **Categorical Encoding** | ✅ | Implemented automated `OneHotEncoder` within a scikit-learn `ColumnTransformer`. |
| **Correlation Heatmap** | ✅ | Analyzed relationships between mechanical parameters (engine size, weight, horsepower) and price. |
| **Train/Test Split** | ✅ | 80/20 train/test split (`random_state=42`). |
| **Multiple Regression Models** | ✅ | Evaluated **Linear Regression**, **Ridge Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**. |
| **Evaluation Metrics** | ✅ | Evaluated Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score. |
| **Feature Importance Chart** | ✅ | Identified the top 10 most influential features driving vehicle market prices. |
| **Interactive Streamlit App** | ✅ | Built an interactive valuation app where users can input custom specs and receive instant valuations. |

---

## 📊 Model Performance Comparison

| Model | $R^2$ Score (%) | RMSE ($) | MAE ($) |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** *(Best Model)* | **95.53%** | **$2,168** | **$1,364** |
| **Gradient Boosting Regressor** | **93.89%** | **$2,534** | **$1,623** |
| **Linear Regression (Baseline)** | 88.58% | $3,456 | $2,425 |
| **Ridge Regression** | 88.62% | $3,450 | $2,410 |

---

## 💡 Key Valuation Insights
* **Engine Size & Horsepower:** The primary drivers of market price, exhibiting strong positive correlation ($r > 0.80$).
* **Fuel Economy Trade-Off:** City and highway MPG correlate negatively with price ($r pprox -0.70$) because luxury high-performance vehicles prioritize power output over fuel efficiency.
* **Top 3 Drivers:** Engine size, curb weight, and horsepower account for over **70% of pricing variance**.

---

## 🛠️ Project Structure

```text
OIBSIP/
└── DataScience-Task3-CarPricePrediction/
    ├── README.md
    ├── CarPrice.csv
    ├── car_price_prediction.ipynb
    ├── train_and_predict.py
    ├── app.py
    └── images/
        ├── price_distribution.png
        ├── price_by_fueltype.png
        ├── price_vs_horsepower.png
        ├── correlation_heatmap.png
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
jupyter notebook car_price_prediction.ipynb
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
  * **Task:** Task 3: Car Price Prediction with Machine Learning
* **Walkthrough:** Explain dataset cleaning and brand typo normalization, show the correlation heatmap and feature importance chart, and test a sample vehicle in the Streamlit app!

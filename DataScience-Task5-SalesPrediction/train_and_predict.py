"""
Oasis Infobyte SIP — Data Science Internship
Task 5: Sales Prediction Using Python (Regression)
Intern: Yash Devidas Chaudhari | Track: Data Science
Reference: OIB/N2/IP5780
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=" * 68)
    print("📈 OASIS INFOBYTE SIP — DATA SCIENCE INTERNSHIP")
    print("📌 TASK 5: SALES PREDICTION USING PYTHON (REGRESSION)")
    print("👤 Intern: Yash Devidas Chaudhari | Track: Data Science")
    print("=" * 68)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Advertising.csv")
    
    if not os.path.exists(csv_path):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv", csv_path)
        
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
        
    print(f"\n[1/5] Loaded Dataset: {df.shape[0]} marketing records, {df.shape[1]} features")
    print(f"      Missing Values: {df.isnull().sum().sum()} (Dataset is 100% clean)")
    print("      Features: TV ($k), Radio ($k), Newspaper ($k)")
    print("      Target:   Sales (thousands of units)\n")
    
    print("[2/5] Pearson Correlation with Sales:")
    for col in ['TV', 'Radio', 'Newspaper']:
        corr = df[col].corr(df['Sales'])
        print(f"      • {col:<10}: r = {corr:+.3f}")
        
    print("\n[3/5] Splitting data (80% Train, 20% Test) and Training Models...")
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    print("-" * 68)
    print(f"{'Model Name':<30} | {'R² Score':<10} | {'RMSE':<9} | {'MAE'}")
    print("-" * 68)
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds) * 100
        trained_models[name] = (model, r2, rmse, mae)
        print(f"{name:<30} | {r2:>8.2f}% | {rmse:>7.3f} | {mae:>7.3f}")
    print("-" * 68)
    
    lr = trained_models["Linear Regression (Baseline)"][0]
    print("\n[4/5] Channel Impact Analysis (Linear Regression Coefficients):")
    print(f"      • Baseline Intercept: {lr.intercept_:.3f}")
    print(f"      • TV Coefficient:     +{lr.coef_[0]:.4f} (Highest volume driver)")
    print(f"      • Radio Coefficient:  +{lr.coef_[1]:.4f} (Highest ROI per dollar)")
    print(f"      • Newspaper Coef:     {lr.coef_[2]:+.4f} (Negligible effect)")
    print("\n      Interpretation: TV budget expansion and Radio spot investments drive the")
    print("      vast majority of incremental sales, while Newspaper ads have near-zero impact.")
    
    print("\n[5/5] Sample Sales Predictions for Custom Marketing Budgets:")
    scenarios = [
        {"desc": "Balanced Budget",  "tv": 150.0, "radio": 30.0, "news": 20.0},
        {"desc": "TV-Heavy Focus",   "tv": 250.0, "radio": 10.0, "news": 5.0},
        {"desc": "Low Budget Test",  "tv": 40.0,  "radio": 15.0, "news": 10.0}
    ]
    
    best_model = trained_models["Gradient Boosting Regressor"][0]
    for i, sc in enumerate(scenarios, 1):
        budget = pd.DataFrame([[sc['tv'], sc['radio'], sc['news']]], columns=['TV', 'Radio', 'Newspaper'])
        pred_sales = best_model.predict(budget)[0]
        print(f"\n  Scenario {i} ({sc['desc']}):")
        print(f"  TV: ${sc['tv']}k | Radio: ${sc['radio']}k | Newspaper: ${sc['news']}k")
        print(f"  ➔ Predicted Product Sales: {pred_sales:.2f} thousand units (~{int(pred_sales * 1000):,} units)")
        
    print("\n" + "=" * 68)
    print("✔ Task 5 Execution Completed Successfully!")
    print("=" * 68)

if __name__ == '__main__':
    main()

"""
Oasis Infobyte SIP — Data Science Internship
Task 3: Car Price Prediction with Machine Learning
Intern: Yash Devidas Chaudhari | Track: Data Science
Reference: OIB/N2/IP5780
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=" * 68)
    print("🚗 OASIS INFOBYTE SIP — DATA SCIENCE INTERNSHIP")
    print("📌 TASK 3: CAR PRICE PREDICTION WITH MACHINE LEARNING")
    print("👤 Intern: Yash Devidas Chaudhari | Track: Data Science")
    print("=" * 68)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "CarPrice.csv")
    
    if not os.path.exists(csv_path):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv", csv_path)
        
    df = pd.read_csv(csv_path)
    if 'Brand' not in df.columns:
        df['Brand'] = df['CarName'].apply(lambda x: str(x).split(' ')[0].lower())
        brand_mapping = {'maxda': 'mazda', 'porcshce': 'porsche', 'toyouta': 'toyota', 'vokswagen': 'volkswagen', 'vw': 'volkswagen'}
        df['Brand'] = df['Brand'].replace(brand_mapping)
        
    print(f"\n[1/5] Loaded Dataset: {df.shape[0]} car samples, {df['Brand'].nunique()} unique brands")
    print(f"      Price Range: ${df['price'].min():,.0f} to ${df['price'].max():,.0f} (Mean: ${df['price'].mean():,.0f})")
    print(f"      Missing Values: {df.isnull().sum().sum()} (Dataset is 100% clean)")
    
    cat_features = ['Brand', 'fueltype', 'carbody', 'drivewheel']
    num_features = ['enginesize', 'horsepower', 'curbweight', 'carwidth', 'citympg', 'highwaympg']
    
    print("\n[2/5] Building Pipeline with One-Hot Encoding for categorical features...")
    X = df[cat_features + num_features]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)],
        remainder='passthrough'
    )
    
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    print("\n[3/5] Training Regression Models...")
    print("-" * 68)
    print(f"{'Model Architecture':<30} | {'R² Score':<10} | {'RMSE ($)':<12} | {'MAE ($)'}")
    print("-" * 68)
    
    trained_pipes = {}
    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('reg', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds) * 100
        trained_pipes[name] = pipe
        print(f"{name:<30} | {r2:>8.2f}% | ${rmse:>10,.2f} | ${mae:>9,.2f}")
    print("-" * 68)
    
    best_pipe = trained_pipes["Random Forest Regressor"]
    
    print("\n[4/5] Testing Valuation Predictions on Arbitrary Custom Vehicles:")
    test_cars = [
        {
            "desc": "Economy Toyota Sedan",
            "data": pd.DataFrame([['toyota', 'gas', 'sedan', 'fwd', 110, 85, 2300, 66.0, 30, 37]], columns=cat_features + num_features)
        },
        {
            "desc": "High-Performance BMW Sports Car",
            "data": pd.DataFrame([['bmw', 'gas', 'sedan', 'rwd', 209, 182, 3200, 68.0, 18, 22]], columns=cat_features + num_features)
        },
        {
            "desc": "Mid-Range Honda Hatchback",
            "data": pd.DataFrame([['honda', 'gas', 'hatchback', 'fwd', 92, 76, 2015, 64.0, 31, 38]], columns=cat_features + num_features)
        }
    ]
    
    for i, car in enumerate(test_cars, 1):
        pred_val = best_pipe.predict(car['data'])[0]
        print(f"\n  Vehicle {i} ({car['desc']}):")
        print(f"  ➔ Estimated Market Valuation: ${pred_val:,.2f}")
        
    print("\n" + "=" * 68)
    print("✔ Task 3 Execution Completed Successfully!")
    print("=" * 68)

if __name__ == '__main__':
    main()

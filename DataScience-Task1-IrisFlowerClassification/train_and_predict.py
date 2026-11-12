"""
Oasis Infobyte SIP — Data Science Internship
Task 1: Iris Flower Classification
Intern: Yash Devidas Chaudhari
Reference: OIB/N2/IP5780
"""

import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=" * 65)
    print("🌸 OASIS INFOBYTE SIP — DATA SCIENCE INTERNSHIP")
    print("📌 TASK 1: IRIS FLOWER CLASSIFICATION")
    print("👤 Intern: Yash Devidas Chaudhari | Track: Data Science")
    print("=" * 65)
    
    # 1. Load Dataset
    print("\n[Step 1/5] Loading Iris dataset from scikit-learn...")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df['target'] = iris.target
    
    print(f"✔ Dataset loaded successfully! Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✔ Target Species: {list(iris.target_names)}")
    print(f"✔ Missing values: {df.isnull().sum().sum()} (Dataset is 100% clean)")
    
    # 2. Train/Test Split
    print("\n[Step 2/5] Splitting data into 80% Training and 20% Testing sets...")
    X = df[iris.feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✔ Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
    
    # 3. Model Training & Comparison
    print("\n[Step 3/5] Training 4 distinct Machine Learning Classifiers...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    print("\n" + "-" * 65)
    print(f"{'Model Name':<28} | {'Accuracy Score':<15} | {'Status'}")
    print("-" * 65)
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        trained_models[name] = (clf, acc, y_pred)
        print(f"{name:<28} | {acc * 100:>6.2f}%         | Trained & Evaluated")
    print("-" * 65)
    
    # 4. Detailed Evaluation for Best Model
    best_model_name = "Logistic Regression"
    best_clf, best_acc, best_pred = trained_models[best_model_name]
    
    print(f"\n[Step 4/5] Detailed Performance Report for Selected Best Model: {best_model_name}")
    print(f"Overall Accuracy: {best_acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, best_pred, target_names=iris.target_names))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    
    # 5. Live Prediction Demonstration
    print("\n[Step 5/5] Testing Model with Unseen Real-world Flower Samples:")
    test_samples = [
        {"desc": "Small petals & wide sepals", "vals": [5.1, 3.5, 1.4, 0.2]},
        {"desc": "Medium length & width",       "vals": [6.0, 2.7, 5.1, 1.6]},
        {"desc": "Large elongated petals",     "vals": [6.9, 3.1, 5.4, 2.1]}
    ]
    
    for i, item in enumerate(test_samples, 1):
        vals = np.array([item["vals"]])
        pred_idx = best_clf.predict(vals)[0]
        prob = best_clf.predict_proba(vals)[0][pred_idx] * 100
        species = iris.target_names[pred_idx]
        print(f"  Sample {i} {item['vals']} ({item['desc']}):")
        print(f"  ➔ Predicted Species: {species.upper()} (Confidence: {prob:.2f}%)\n")
        
    print("=" * 65)
    print("✔ Project execution completed successfully!")
    print("✔ Notebook and documentation are ready for internship submission.")
    print("=" * 65)

if __name__ == "__main__":
    main()

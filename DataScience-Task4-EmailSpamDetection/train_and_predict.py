"""
Oasis Infobyte SIP — Data Science Internship
Task 4: Email / SMS Spam Detection with Machine Learning (NLP)
Intern: Yash Devidas Chaudhari | Track: Data Science
Reference: OIB/N2/IP5780
"""

import os
import re
import string
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def clean_text(text, stop_words):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def main():
    print("=" * 68)
    print("📧 OASIS INFOBYTE SIP — DATA SCIENCE INTERNSHIP")
    print("📌 TASK 4: EMAIL SPAM DETECTION WITH NLP & MACHINE LEARNING")
    print("👤 Intern: Yash Devidas Chaudhari | Track: Data Science")
    print("=" * 68)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "spam.csv")
    
    if not os.path.exists(csv_path):
        print("Dataset not found locally, downloading...")
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv", csv_path)
    
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        df.columns = ['label', 'message']
    if 'target' not in df.columns:
        df['target'] = df['label'].map({'ham': 0, 'spam': 1})
        
    print(f"\n[1/5] Loaded Dataset: {df.shape[0]} total messages")
    print(f"      Ham (Legitimate): {sum(df['target'] == 0)} ({sum(df['target'] == 0)/len(df)*100:.1f}%)")
    print(f"      Spam:             {sum(df['target'] == 1)} ({sum(df['target'] == 1)/len(df)*100:.1f}%)")
    
    print("\n[2/5] Preprocessing text pipeline (lowercasing, punctuation, stopwords)...")
    stop_words = set(stopwords.words('english'))
    df['cleaned'] = df['message'].apply(lambda t: clean_text(str(t), stop_words))
    
    print("[3/5] Extracting TF-IDF features (Term Frequency - Inverse Document Frequency)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['target'], test_size=0.2, random_state=42, stratify=df['target']
    )
    
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    print(f"      Vocabulary size: {X_train_vec.shape[1]} n-gram features")
    
    print("\n[4/5] Training Classifiers...")
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
        "Linear Support Vector (SVM)": LinearSVC(random_state=42, max_iter=2000)
    }
    
    print("-" * 68)
    print(f"{'Classifier Model':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score'}")
    print("-" * 68)
    
    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds) * 100
        prec = precision_score(y_test, preds) * 100
        rec = recall_score(y_test, preds) * 100
        f1 = f1_score(y_test, preds) * 100
        print(f"{name:<28} | {acc:>7.2f}% | {prec:>7.2f}% | {rec:>7.2f}% | {f1:>7.2f}%")
    print("-" * 68)
    
    best_clf = models["Multinomial Naive Bayes"]
    print("\nDetailed Evaluation for Multinomial Naive Bayes (Industry Standard for Text):")
    nb_preds = best_clf.predict(X_test_vec)
    print(classification_report(y_test, nb_preds, target_names=['Ham', 'Spam']))
    
    print("\n[5/5] Live Demonstration with New Custom Email/SMS Samples:")
    test_emails = [
        "Congratulations! You've won a $1,000 Walmart gift card. Click here now to claim your prize! Call 080012345.",
        "Hi Yash, please remember to bring the project presentation slides to tomorrow morning's team meeting.",
        "URGENT: Your bank account has been suspended due to suspicious activity. Verify your PIN immediately at http://secure-login.xyz",
        "Hey, are we still meeting up for coffee this afternoon? Let me know!"
    ]
    
    for i, email in enumerate(test_emails, 1):
        cleaned = clean_text(email, stop_words)
        vec = tfidf.transform([cleaned])
        pred = best_clf.predict(vec)[0]
        prob = best_clf.predict_proba(vec)[0]
        label = "🚨 SPAM" if pred == 1 else "✅ HAM (Legitimate)"
        conf = prob[pred] * 100
        print(f"\n  Message {i}: \"{email[:75]}...\"")
        print(f"  Result: {label} (Confidence: {conf:.2f}%)")
        
    print("\n" + "=" * 68)
    print("✔ Task 4 Execution Completed Successfully!")
    print("=" * 68)

if __name__ == '__main__':
    main()

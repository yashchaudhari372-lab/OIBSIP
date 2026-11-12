# 📧 Task 4: Email Spam Detection with Machine Learning

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oibsip-spam-detection-yash.streamlit.app)  
🌐 **Live Web Application:** [https://oibsip-spam-detection-yash.streamlit.app](https://oibsip-spam-detection-yash.streamlit.app)

**Oasis Infobyte — SIP (Student Internship Program)**  
* **Intern Name:** Yash Devidas Chaudhari  
* **Track:** Data Science  
* **Offer Letter ID:** OIB/N2/IP5780  
* **Repository:** `OIBSIP`  
* **Folder Name:** `DataScience-Task4-EmailSpamDetection`  
* **Live Link:** https://oibsip-spam-detection-yash.streamlit.app/  

---

## 📌 Project Overview
Spam communications (unsolicited commercial emails, phishing schemes, lottery scams) present serious security and privacy threats. The objective of this project is to develop a Natural Language Processing (NLP) classification pipeline that accurately classifies incoming communications as **HAM** (legitimate) or **SPAM**.

---

## 📋 Feature Checklist Compliance

| Requirement from Task Card | Status | Implementation Details |
| :--- | :---: | :--- |
| **Download Dataset** | ✅ | SMS Spam Collection dataset (5,572 labeled messages) bundled locally as `spam.csv`. |
| **Class Distribution Check** | ✅ | Analyzed counts: 4,825 Ham (86.6%) vs. 747 Spam (13.4%). |
| **Text Preprocessing** | ✅ | Applied lowercasing, URL removal, punctuation stripping, and NLTK stopword filtering. |
| **TF-IDF Feature Extraction** | ✅ | Implemented `TfidfVectorizer(max_features=3000, ngram_range=(1, 2))` capturing unigram and bigram token statistics. |
| **Train/Test Split** | ✅ | Stratified 80/20 split (`random_state=42`) preserving the original class distribution. |
| **Train $\ge$ 2 Classifiers** | ✅ | Trained **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear SVM**. |
| **Comprehensive Evaluation** | ✅ | Evaluated Accuracy, Precision, Recall, F1-Score, and Confusion Matrix across all models. |
| **Recall Discussion** | ✅ | Detailed analysis of the precision-recall trade-off and why Recall is vital in spam/phishing security. |
| **Bonus WordClouds** | ✅ | Generated high-resolution WordClouds highlighting top spam keywords (`free`, `claim`, `prize`, `call`). |
| **Interactive Web App** | ✅ | Streamlit web application (`app.py`) for live message testing with probability breakdowns. |

---

## 📊 Model Performance Comparison

| Classifier | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **98.83%** | **97.90%** | **93.29%** | **0.955** |
| **Linear Support Vector (SVM)** | **98.74%** | **96.55%** | **93.96%** | **0.952** |
| **Logistic Regression** | **97.31%** | **98.39%** | **81.88%** | **0.894** |

### Why Multinomial Naive Bayes is Selected:
1. **Industry Standard:** Naive Bayes is computationally lightweight and excels at text classification by computing independent feature probabilities.
2. **Balanced Performance:** Delivers **98.83% accuracy** with an impressive **93.29% recall** on spam messages.

---

## 🛠️ Project Structure

```text
OIBSIP/
└── DataScience-Task4-EmailSpamDetection/
    ├── README.md
    ├── spam.csv
    ├── spam_detection.ipynb
    ├── train_and_predict.py
    ├── app.py
    └── images/
        ├── class_distribution.png
        ├── wordcloud_spam.png
        ├── wordcloud_ham.png
        ├── message_length_distribution.png
        ├── confusion_matrices.png
        └── model_comparison.png
```

---

## 🚀 How to Run Locally

### 1. Terminal Run (Instant Evaluation & Live Test)
```bash
python train_and_predict.py
```

### 2. Interactive Jupyter Notebook
```bash
jupyter notebook spam_detection.ipynb
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
  * **Task:** Task 4: Email Spam Detection with Machine Learning
* **Walkthrough:** Show the dataset balance, WordClouds, TF-IDF explanation, model comparison bar chart, and test a live spam message!

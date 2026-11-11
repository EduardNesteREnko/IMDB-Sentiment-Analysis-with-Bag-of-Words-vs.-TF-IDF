# IMDB Sentiment Analysis with Bag-of-Words vs. TF-IDF

This repository contains a Python machine learning script for text classification and sentiment analysis on the **IMDB Dataset of 50K Movie Reviews**. The script compares two standard text vectorization techniques—**Bag-of-Words (CountVectorizer)** and **TF-IDF (TfidfVectorizer)**—using **Logistic Regression**.

## 📋 Pipeline Overview

1. **Data Loading & Preprocessing**:
   * Loads the IMDB dataset from a CSV file.
   * Maps textual sentiment labels (`positive` / `negative`) to binary integers (`1` and `0`).
   * Subsamples $20,000$ reviews with a fixed random state for efficient execution.

2. **Train-Test Split**:
   * Splits the processed text and labels into training ($80\%$) and testing ($20\%\$) sets.

3. **Text Vectorization**:
   * **Bag-of-Words**: Transforms text using `CountVectorizer` (capped at $10,000$ max features with English stop words removed).
   * **TF-IDF**: Transforms text using `TfidfVectorizer` (accounting for term frequency-inverse document frequency, also capped at $10,000$ features).

4. **Model Training & Evaluation**:
   * Fits a `LogisticRegression` classifier on both vectorized representations.
   * Compares model performance using **Accuracy Score** and a detailed **Classification Report**.

---

## 🛠️ Code Implementation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare dataset
df = pd.read_csv('IMDB Dataset.csv')
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
df = df.sample(n=20000, random_state=42).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# 1. Bag-of-Words (CountVectorizer)
count_vec = CountVectorizer(max_features=10000, stop_words='english')
X_train_bow = count_vec.fit_transform(X_train)
X_test_bow = count_vec.transform(X_test)

model_bow = LogisticRegression(max_iter=1000, random_state=42)
model_bow.fit(X_train_bow, y_train)

y_pred_bow = model_bow.predict(X_test_bow)
print(f"Accuracy (BoW): {accuracy_score(y_test, y_pred_bow):.4f}")

# 2. TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

model_tfidf = LogisticRegression(max_iter=1000, random_state=42)
model_tfidf.fit(X_train_tfidf, y_train)

y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
print(f"Accuracy (TF-IDF): {accuracy_score(y_test, y_pred_tfidf):.4f}")

print("\nClassification Report (TF-IDF):")
print(classification_report(y_test, y_pred_tfidf, target_names=['Negative', 'Positive']))
```

---

## 📊 Results & Observations

* **TF-IDF vs. BoW**: Typically, TF-IDF scaling helps penalize overly frequent words that carry less semantic weight, usually yielding competitive or superior performance compared to raw frequency counts in Bag-of-Words.
* **Logistic Regression**: Serves as a strong, highly interpretable linear baseline for text classification tasks when paired with sparse matrix features.
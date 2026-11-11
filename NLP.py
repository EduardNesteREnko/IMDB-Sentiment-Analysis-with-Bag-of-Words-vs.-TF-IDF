import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('IMDB Dataset.csv')
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
df = df.sample(n=20000, random_state=42).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

count_vec = CountVectorizer(max_features=10000, stop_words='english')
X_train_bow = count_vec.fit_transform(X_train)
X_test_bow = count_vec.transform(X_test)

model_bow = LogisticRegression(max_iter=1000, random_state=42)
model_bow.fit(X_train_bow, y_train)

y_pred_bow = model_bow.predict(X_test_bow)
print(f"Accuracy (BoW): {accuracy_score(y_test, y_pred_bow):.4f}")

tfidf_vec = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

model_tfidf = LogisticRegression(max_iter=1000, random_state=42)
model_tfidf.fit(X_train_tfidf, y_train)

y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
print(f"Accuracy (TF-IDF): {accuracy_score(y_test, y_pred_tfidf):.4f}")

print("\nClassification Report (TF-IDF):")
print(classification_report(y_test, y_pred_tfidf, target_names=['Negative', 'Positive']))
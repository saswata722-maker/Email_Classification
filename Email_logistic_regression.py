from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# 1. Load dataset (data/spam_ham_dataset.csv next to this script; see data/README.md)
path = Path(__file__).resolve().parent / "data" / "spam_ham_dataset.csv"
df = pd.read_csv(path)

# 2. Preprocess text
if "text" in df.columns:
    df["text"] = df["text"].fillna("").astype(str).str.strip()
else:
    subj = df["Subject"].fillna("") if "Subject" in df.columns else ""
    msg = df["Message"].fillna("") if "Message" in df.columns else ""
    df["text"] = (subj + " " + msg).str.strip()

# Drop empty messages and duplicates
df = df[df["text"] != ""].drop_duplicates(subset=["text"])

# 3. Encode labels (spam=1, ham=0)
if "label_num" in df.columns:
    df["label"] = df["label_num"].astype(int)
else:
    label_col = "Spam/Ham" if "Spam/Ham" in df.columns else "label"
    label_map = {"ham": 0, "spam": 1, "0": 0, "1": 1}
    df["label"] = df[label_col].astype(str).str.strip().str.strip('"\'').str.lower().map(label_map)

df = df.dropna(subset=["label"])

X = df["text"]
y = df["label"].astype(int)

print(f"Total samples: {len(df)} (Ham: {(y == 0).sum()}, Spam: {(y == 1).sum()})")

# 4. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Build and Train Pipeline
model = make_pipeline(
    TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        min_df=2,
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    ),
    LogisticRegression(max_iter=2000, class_weight="balanced")
)

print("Training model...")
model.fit(X_train, y_train)

# 6. Evaluate Model
pred = model.predict(X_test)

print("\n" + "=" * 55)
print("             MODEL EVALUATION METRICS")
print("=" * 55)
print(f"Accuracy Score:      {accuracy_score(y_test, pred):.4f} ({accuracy_score(y_test, pred):.2%})")
print(f"F1-Score (Spam):     {f1_score(y_test, pred, pos_label=1):.4f}")
print(f"F1-Score (Ham):      {f1_score(y_test, pred, pos_label=0):.4f}")
print(f"F1-Score (Macro):    {f1_score(y_test, pred, average='macro'):.4f}")
print(f"F1-Score (Weighted): {f1_score(y_test, pred, average='weighted'):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, pred)
print(f"  TN (Ham correctly classified):  {cm[0][0]}")
print(f"  FP (Ham misclassified as Spam): {cm[0][1]}")
print(f"  FN (Spam misclassified as Ham): {cm[1][0]}")
print(f"  TP (Spam correctly classified): {cm[1][1]}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, pred, target_names=["Ham", "Spam"], digits=4))
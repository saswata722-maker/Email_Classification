# 📧 Email Spam Classifier (TF-IDF + Classical ML Models)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)

An end-to-end Machine Learning comparison for classifying emails as **Spam** or **Ham** (legitimate). Five TF-IDF pipelines (Logistic Regression, Naive Bayes, Random Forest, SVM, XGBoost) are trained and evaluated on the Spam Mails Dataset (`spam_ham_dataset.csv`), with a shared 80/20 stratified split so results are directly comparable.

---

## 🚀 Key Features

- **Robust Text Preprocessing**: Uses the `text` column directly (falls back to combining Subject + Message for Enron-style files), cleans whitespace, strips quotes, and removes exact-duplicate messages.
- **Advanced TF-IDF Vectorization** (shared by all five models):
  - Word unigrams and bigrams (`ngram_range=(1, 2)`)
  - Sublinear term frequency scaling (`sublinear_tf=True`)
  - English stop-words filtering and `min_df=2`
  - Vocabulary capped at `50,000` features to balance vocabulary depth and memory efficiency
- **Five Classifiers** (same split, `random_state=42`):
  - Logistic Regression (`class_weight="balanced"`, `max_iter=2000`)
  - Multinomial Naive Bayes (`alpha=0.1`)
  - Random Forest (`n_estimators=300`, `class_weight="balanced"`)
  - Linear SVM (`LinearSVC`, `C=1.0`, `class_weight="balanced"`)
  - XGBoost (`n_estimators=300`, `max_depth=6`, `learning_rate=0.1`)
- **Comprehensive Evaluation Suite**:
  - Accuracy, Spam F1, Ham F1, Macro F1, and Weighted F1.
  - Formatted 2x2 Confusion Matrix breakdown (TN, FP, FN, TP).
  - Detailed Precision, Recall, and F1 classification report per class.

---

## 📁 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI workflow
├── data/
│   └── README.md                    # Dataset instructions & format guide
├── .gitattributes                   # Git LF line-ending normalization
├── .gitignore                       # Ignored files (models, virtualenvs, datasets)
├── Email_logistic_regression.py     # Logistic Regression training, evaluation & inference
├── Email_naive_bayes.py           # Naive Bayes training & evaluation
├── Email_random_forest.py         # Random Forest training & evaluation
├── Email_SVM.py                   # SVM (LinearSVC) training & evaluation
├── Email_xgboost.py               # XGBoost training & evaluation
├── LICENSE                          # MIT Open Source License
├── README.md                        # Project documentation
└── requirements.txt                 # Python package dependencies
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

Download [`spam_ham_dataset.csv` from Kaggle (Spam Mails Dataset by venky73)](https://www.kaggle.com/datasets/venky73/spam-mails-dataset) and place it in the `data/` directory as `data/spam_ham_dataset.csv` (see [`data/README.md`](data/README.md)). The file is git-ignored, so each user downloads it locally. The scripts load it via a path relative to the repo (`Path(__file__).parent / "data" / "spam_ham_dataset.csv"`), so no absolute-path editing is needed.

The dataset has 5,171 rows (3,672 Ham / 1,499 Spam; ~4,993 after exact-duplicate removal), with columns `text`, `label` (`ham`/`spam`) and `label_num` (`0`/`1`).

The scripts also accept Enron-style files with `Subject` + `Message` text columns and a `Spam/Ham` (or `label`) label column.

---

## 💻 Usage

Run any of the five training/evaluation scripts (each needs `data/spam_ham_dataset.csv` in place):
```bash
python Email_logistic_regression.py
python Email_naive_bayes.py
python Email_random_forest.py
python Email_SVM.py
python Email_xgboost.py
```

Each script will:
1. Load `data/spam_ham_dataset.csv`.
2. Preprocess text (use `text`, or combine Subject & Message for Enron-style files; remove empty entries, drop exact duplicates).
3. Encode labels (`ham` $\rightarrow$ 0, `spam` $\rightarrow$ 1).
4. Perform an 80/20 stratified train/test split (`random_state=42`).
5. Train its TF-IDF + classifier pipeline.
6. Print comprehensive evaluation metrics (Accuracy, F1-Scores, Confusion Matrix, Classification Report).

---

## 📊 Model Comparison

All five models were run on the same dataset (4993 samples — 3531 Ham / 1462 Spam, 80/20 stratified split, `random_state=42`) with identical TF-IDF settings. Results below are the live outputs of each script:

| Model | Script | Accuracy | F1 (Spam) | F1 (Ham) | F1 (Macro) | F1 (Weighted) |
|---|---|---|---|---|---|---|
| Logistic Regression | `Email_logistic_regression.py` | 0.9670 (96.70%) | 0.9467 | 0.9761 | 0.9614 | 0.9675 |
| Naive Bayes (MultinomialNB) | `Email_naive_bayes.py` | 0.9680 (96.80%) | 0.9472 | 0.9770 | 0.9621 | 0.9683 |
| Random Forest | `Email_random_forest.py` | 0.9680 (96.80%) | 0.9481 | 0.9768 | 0.9624 | 0.9684 |
| XGBoost | `Email_xgboost.py` | 0.9690 (96.90%) | 0.9484 | 0.9778 | 0.9631 | 0.9692 |
| SVM (LinearSVC) | `Email_SVM.py` | **0.9890 (98.90%)** | **0.9815** | **0.9922** | **0.9868** | **0.9890** |

> 🏆 **Best model: SVM (LinearSVC)** — highest accuracy and highest F1 across all averages.
> Reproduce with: `python Email_logistic_regression.py`, `python Email_naive_bayes.py`, `python Email_random_forest.py`, `python Email_xgboost.py`, `python Email_SVM.py`
>
> ⚠️ **Leakage caveat**: only *exact*-duplicate texts are removed, so *near*-duplicate emails can still appear in both train and test splits, which may inflate scores (SVM's 98.9% especially). Treat these numbers as optimistic until validated with grouped/de-duplicated splits.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

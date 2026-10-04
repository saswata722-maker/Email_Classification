# 📧 Email Spam Classifier (TF-IDF & Logistic Regression)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)

An end-to-end, high-performance Machine Learning pipeline for classifying emails as **Spam** or **Ham** (legitimate). Powered by **TF-IDF Vectorization** (with sublinear term frequency scaling and bi-grams) and **Balanced Logistic Regression**, this model delivers fast training and high precision/recall on imbalanced email corpora like the Enron Spam Dataset.

---

## 🚀 Key Features

- **Robust Text Preprocessing**: Combines subject lines and message bodies, cleans whitespace, strips quotes, and removes duplicate messages.
- **Advanced TF-IDF Vectorization**:
  - Word unigrams and bigrams (`ngram_range=(1, 2)`)
  - Sublinear term frequency scaling (`sublinear_tf=True`)
  - English stop-words filtering and `min_df=2`
  - Vocabulary capped at `50,000` features to balance vocabulary depth and memory efficiency
- **Class-Balanced Logistic Regression**:
  - Automatically adjusts weights inversely proportional to class frequencies (`class_weight="balanced"`).
  - High iteration convergence limit (`max_iter=2000`).
- **Comprehensive Evaluation Suite**:
  - Accuracy, Spam F1, Ham F1, Macro F1, and Weighted F1.
  - Formatted 2x2 Confusion Matrix breakdown (TN, FP, FN, TP).
  - Detailed Precision, Recall, and F1 classification report per class.
- **Model Serialization & CLI Inference**:
  - Save trained pipelines with `joblib` for deployment.
  - Test custom email messages directly from the command line.

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
├── Email_logistic_regression.py     # Main training, evaluation & inference script
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

Place your dataset CSV in the `data/` directory as `data/enron_spam_data.csv`, or specify a custom path with `--data-path`.

The CSV file must contain the following columns:
- **`Subject`**: Subject line of the email
- **`Message`** (or `text`): Body text of the email
- **`Spam/Ham`** (or `label`): Label values (`spam` / `ham` or `1` / `0`)

> 💡 **Dataset Download**: You can get the Enron Spam dataset from [Kaggle](https://www.kaggle.com/) or public machine learning dataset repositories.

---

## 💻 Usage

Run the training and evaluation script directly:
```bash
python Email_logistic_regression.py
```

The script will:
1. Load the Enron email dataset.
2. Preprocess text (combine Subject & Message, remove empty entries, drop duplicates).
3. Encode labels (`ham` $\rightarrow$ 0, `spam` $\rightarrow$ 1).
4. Perform an 80/20 stratified train/test split.
5. Train a TF-IDF + Logistic Regression pipeline.
6. Print comprehensive evaluation metrics (Accuracy, F1-Scores, Confusion Matrix, Classification Report).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

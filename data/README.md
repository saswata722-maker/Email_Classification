# 📊 Dataset Setup & Instructions

This directory is intended to store your email spam dataset CSV files. Large dataset files are ignored by git via `.gitignore` to keep the repository lightweight and prevent file size limits on GitHub.

---

## 🥇 Recommended Dataset: `spam_ham_dataset.csv`

For real-world accuracy without duplicate text bias, we recommend the **Spam Mails Dataset** from Kaggle:

* **Source**: [Kaggle: Spam Mails Dataset (by venky73)](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
* **File Name**: `spam_ham_dataset.csv`
* **Total Samples**: 5,172 emails (3,672 Ham, 1,499 Spam — ~4,993 after deduplication)
* **Columns**:
  | Column | Type | Description |
  | :--- | :--- | :--- |
  | `text` | string | Full email text including Subject line and Message body |
  | `label_num` | integer | Target label: `0` = Ham (legitimate), `1` = Spam |
  | `label` | string | Text label: `'ham'` or `'spam'` |

---

## 📂 Supported Dataset Schemas

The training pipeline in [`Email_logistic_regression.py`](../Email_logistic_regression.py) automatically detects and handles multiple CSV column formats:

### Format A: Unified Text Column *(e.g. `spam_ham_dataset.csv`)*
* **Text**: `text`
* **Label**: `label_num` (`0` / `1`) or `label` (`'ham'` / `'spam'`)

### Format B: Split Subject and Message *(e.g. `enron_spam_data.csv`)*
* **Text**: Combined automatically from `Subject` and `Message`
* **Label**: `Spam/Ham` or `label`

---

## 🚀 How to Set Up Your Dataset

1. Download [`spam_ham_dataset.csv`](https://www.kaggle.com/datasets/venky73/spam-mails-dataset) from Kaggle.
2. Place the CSV file into this `data/` folder:
   ```text
   data/
   └── spam_ham_dataset.csv
   ```
3. Update the `path` variable in [`Email_logistic_regression.py`](../Email_logistic_regression.py):
   ```python
   path = r"data/spam_ham_dataset.csv"
   ```
4. Run the script:
   ```bash
   python Email_logistic_regression.py
   ```

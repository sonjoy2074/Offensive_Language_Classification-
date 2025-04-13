# Offensive_Language_Classification-

## 🔍 Project Overview

This project focuses on building a multi-label classification model to detect various forms of offensive language in user feedback. A single comment may exhibit one or more types of offensiveness such as toxicity, vulgarity, threats, etc. The aim is to accurately identify these categories using NLP and machine learning techniques, including transformer-based models like BERT.

---

## 📂 Dataset Description

The dataset consists of labeled feedback entries with the following columns:

- `id`: Unique identifier for each comment
- `feedback_text`: The comment to be classified
- Offensive categories (binary labels: 0 or 1):
  - `toxic`
  - `abusive`
  - `vulgar`
  - `menace`
  - `offense`
  - `bigotry`

Each comment may have none, one, or multiple offensive labels. There is also an unlabeled `test.csv` file for prediction.

---

## 🛠️ Model Implementation

### 1. Logistic Regression (Baseline)
- **Text Vectorization**: TF-IDF
- **Model**: Multi-label classification using `OneVsRestClassifier`
- **Purpose**: Provides a simple and interpretable benchmark to evaluate against deep learning models

### 2. LSTM (Sequential Deep Learning)
- **Embedding**: Keras `Embedding` layer or pre-trained GloVe embeddings
- **Architecture**: Bi-directional LSTM with dropout and dense layers for multi-label output
- **Loss Function**: Binary Cross-Entropy (for multi-label classification)
- **Optimizer**: Adam
- **Framework**: TensorFlow/Keras
- **Purpose**: Captures sequential patterns and context better than traditional models

### 3. BERT (Transformer-Based)
- **Pre-trained Model**: `bert-base-uncased`
- **Embedding & Fine-tuning**: Leveraged BERT's attention mechanism to fine-tune on the feedback text
- **Model**: `BertForSequenceClassification` with `num_labels=6` and `problem_type="multi_label_classification"`
- **Training**: Managed with Hugging Face `Trainer` API
- **Purpose**: Achieves state-of-the-art performance by understanding deep contextual relationships

---

## 🧼 Text Preprocessing

- Lowercasing
- Tokenization
- Stopword and punctuation removal
- Lemmatization
- Feature extraction:
  - TF-IDF (for logistic regression)
  - BERT tokenizer and embeddings (for BERT model)

---

## 📊 Evaluation Metrics

Evaluation is done using the following metrics on the validation set:

- **Accuracy**
- **Precision, Recall, and F1-score** (for each class)
- **AUC-ROC Curve** (overall performance)
- **Confusion Matrix** (to analyze misclassification patterns)

---


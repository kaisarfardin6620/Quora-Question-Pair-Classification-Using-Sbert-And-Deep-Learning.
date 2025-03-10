# Quora-Question-Pair-Classification-Using-Sbert-And-Deep-Learning.


## 📌 Project Overview
This project classifies Quora question pairs as either **"duplicate"** (i.e., semantically similar) or **"not duplicate"** using **pretrained SBERT embeddings** and multiple machine learning models, including ANN, LSTM, and Siamese Networks.

## 📂 Dataset Information
The dataset consists of labeled question pairs from Quora.

- `train.csv` → Contains labeled question pairs with:
  - `question1` & `question2` (text data)
  - `is_duplicate` (target variable: 1 = duplicate, 0 = not duplicate)
- `test.csv` → Contains question pairs for prediction.
- `sample_submission.csv` → Example format for final submission.

## 🛠️ Steps Followed

### 🔹 1. Exploratory Data Analysis (EDA)
- Checked for missing values and class distribution.
- Analyzed question lengths, word distributions, and duplicate patterns.

### 🔹 2. Preprocessing & Feature Engineering
- Used **SBERT (Sentence-BERT)** to generate high-quality sentence embeddings.
- **Removed outliers** (very short or very long questions).
- Applied **SMOTE** to balance the dataset (if needed).

### 🔹 3. Train-Test Split
- Split the dataset into **train (80%)** and **validation (20%)** for fair evaluation.

### 🔹 4. Model Training
We trained multiple models using **SBERT embeddings** as input:
1. **Traditional ML Models** (Logistic Regression, SVM, Random Forest)
2. **Deep Learning Models** (ANN, LSTM)
3. **Siamese Network** for text similarity detection.

### 🔹 5. Model Evaluation & Visualization
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix** (Heatmap for error analysis)
- **ROC-AUC Curve** (Model performance comparison)
- **Overfitting Check** (Training vs. Validation loss)

### 🔹 6. Final Submission File Generation
- Made predictions on the test dataset.
- Created `final_submission.csv` in the correct format.

## 🚀 How to Run the Notebook
1. **Open Google Colab** and upload the notebook (`.ipynb` file).
2. **Upload the dataset (`train.csv`, `test.csv`)** to the Colab environment.
3. **Run all cells** step by step.
4. The final submission file (`final_submission.csv`) will be generated.

## 📢 Dependencies & Libraries Used
```bash
pip install sentence-transformers
pip install imbalanced-learn
pip install seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Embedding
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer

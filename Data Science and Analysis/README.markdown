# Machine Learning Projects

## Overview
This repository contains four machine learning projects demonstrating exploratory data analysis (EDA), classification, and regression tasks using real-world datasets from Kaggle. The projects include EDA and visualization of the Titanic dataset, sentiment analysis of IMDB movie reviews, fraud detection in credit card transactions, and house price prediction using the Boston Housing dataset. Due to their size, the datasets are hosted on Google Drive.

## Dataset Access
The datasets used in these projects are too large to be hosted directly on GitHub. You can download them from the following Google Drive folder:
- **Download Datasets**: [Google Drive Link](https://drive.google.com/drive/folders/1T7gcRtNz0nPoV7MjVeAoNHw520s0BOmy?usp=drive_link)

The folder contains:
- **Titanic**: `train.csv`, `test.csv`, `gender_submission.csv`
- **IMDB Movie Reviews**: `IMDB Dataset.csv`
- **Credit Card Fraud Detection**: `creditcard.csv`
- **Boston Housing**: `housing.csv`

## Projects

### Task 1: EDA and Visualization of Titanic Dataset
- **Dataset**: Titanic (Kaggle)
- **Description**: Performed exploratory data analysis (EDA) to uncover patterns and insights, including data cleaning, visualization, and documentation of findings.
- **Steps**:
  - Loaded dataset using Pandas.
  - Cleaned data: handled missing values, removed duplicates, treated outliers.
  - Created visualizations:
    - Bar charts for categorical variables (e.g., Sex, Pclass).
    - Histograms for numerical variables (e.g., Age, Fare).
    - Correlation heatmap to explore feature relationships.
  - Documented key insights and patterns (e.g., survival rates by class, gender).
- **Dataset Details**:
  - Source: [Kaggle Titanic](https://www.kaggle.com/c/titanic/data)
  - Size: ~90 KB (train.csv: 891 rows, 12 columns)
  - Files: `train.csv`, `test.csv`, `gender_submission.csv`

### Task 2: Text Sentiment Analysis of IMDB Movie Reviews
- **Dataset**: IMDB Movie Reviews (Kaggle)
- **Description**: Built a sentiment classification model to predict whether movie reviews are positive or negative.
- **Steps**:
  - Preprocessed text: tokenization, stopword removal, lemmatization.
  - Converted text to numerical format using TF-IDF.
  - Trained classifiers: Logistic Regression, Naive Bayes.
  - Evaluated performance using precision, recall, and F1-score.
- **Dataset Details**:
  - Source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  - Size: ~66 MB (50,000 rows, 2 columns)
  - File: `IMDB Dataset.csv`

### Task 3: Fraud Detection System for Credit Card Transactions
- **Dataset**: Credit Card Fraud Detection (Kaggle)
- **Description**: Developed a binary classification model to detect fraudulent transactions, addressing class imbalance and building a CLI for testing.
- **Steps**:
  - Handled class imbalance using SMOTE and undersampling.
  - Trained models: Random Forest, Gradient Boosting.
  - Evaluated model with precision, recall, and F1-score.
  - Built a simple command-line interface (CLI) to test inputs.
- **Dataset Details**:
  - Source: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  - Size: ~150 MB (284,807 rows, 31 columns)
  - File: `creditcard.csv`

### Task 4: House Price Prediction with Boston Housing Dataset
- **Dataset**: Boston Housing (Kaggle)
- **Description**: Built and compared regression models to predict house prices, with feature importance analysis for tree-based models.
- **Steps**:
  - Preprocessed data and features.
  - Implemented models from scratch: Linear Regression, Random Forest, XGBoost.
  - Compared models using RMSE and R² metrics.
  - Visualized feature importance for tree-based models.
- **Dataset Details**:
  - Source: [Kaggle Boston Housing](https://www.kaggle.com/datasets/crawford/boston-housing)
  - Size: ~36 KB (506 rows, 14 columns)
  - File: `housing.csv`

## Observations & Highlights
- Gained expertise in cleaning, analyzing, and visualizing real-world data.
- Developed a deeper understanding of machine learning by building models from scratch.
- Evaluated and optimized model performance using industry-standard metrics (e.g., precision, recall, F1-score, RMSE, R²).
- Acquired experience in specialized domains: text analysis (sentiment) and anomaly detection (fraud).

## Prerequisites
To run the project code (if included), install the following Python libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn nltk
```

## How to Use
1. **Download Datasets**:
   - Access the datasets via the [Google Drive link](https://drive.google.com/drive/folders/1T7gcRtNz0nPoV7MjVeAoNHw520s0BOmy?usp=drive_link).
   - Save them to a local folder (e.g., `datasets/`).
2. **Run Code**:
   - If Jupyter Notebooks or scripts are provided in the repository, open them in a Python environment (e.g., Jupyter Notebook, VS Code).
   - Update file paths in the code to point to the downloaded dataset files.
3. **Explore**:
   - Review the code and outputs for EDA, model training, and visualizations.
   - Refer to documented insights in each task for key findings.

## Notes
- **Dataset Size**: The datasets are too large for GitHub, hence hosted on Google Drive. Ensure you have ~300 MB of storage for downloading.
- **Access**: Verify the Google Drive folder contains all expected files (`train.csv`, `test.csv`, `gender_submission.csv`, `IMDB Dataset.csv`, `creditcard.csv`, `housing.csv`).
- **Contributing**: For issues or contributions, open an issue or pull request on this repository.

## Acknowledgments
- Datasets provided by Kaggle.
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `nltk`, `imbalanced-learn`.
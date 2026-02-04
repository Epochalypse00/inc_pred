Classical Machine Learning – Income Prediction (Foundations)
📌 Project Overview

This project focuses on mastering foundational classical machine learning practices using a tabular income prediction task.
The goal is not to maximize benchmark scores, but to deeply understand how real-world ML pipelines are built, evaluated, and interpreted.

The project emphasizes:

leakage-safe preprocessing

correct metric usage

cross-validation

threshold tuning

model comparison and decision trade-offs

This work reflects intern-level ML engineering expectations.

🎯 Problem Statement

Given demographic and employment-related features, predict whether an individual earns:

<= 50K

> 50K

This is a binary classification problem with class imbalance, making it suitable for studying precision–recall trade-offs.

📊 Dataset

Source: Hugging Face

Dataset: scikit-learn/adult-census-income

Derived from the UCI Adult Census dataset

Feature Types

Numeric: age, hours_per_week, capital_gain, etc.

Categorical: education, occupation, marital_status, workclass, etc.

The dataset does not provide a predefined test split, so a manual stratified train/validation split was created.

🧠 Project Objectives

This project was designed to practice and demonstrate:

Correct handling of numeric vs categorical features

Building leakage-safe sklearn Pipelines

Understanding bias, variance, and stability

Using cross-validation instead of single splits

Comparing linear vs non-linear models

Interpreting confusion matrices and error types

Making metric-driven decisions rather than accuracy-driven ones

🛠️ Methodology
1️⃣ Data Splitting

80/20 train–validation split

Stratified by target variable

Fixed random seed for reproducibility

2️⃣ Preprocessing (Leakage-Safe)

All preprocessing steps were performed inside sklearn Pipelines, ensuring that transformations were learned only from training data.

Numeric features

Median imputation

Standard scaling

Categorical features

Most-frequent imputation

One-hot encoding (handle_unknown="ignore")

3️⃣ Models Trained
Logistic Regression

Linear baseline

High interpretability

Strong ranking ability (ROC-AUC)

Conservative classification behavior

Random Forest

Non-linear ensemble model

Captures feature interactions

Higher recall and F1 score

Reduced interpretability compared to linear models

📈 Model Comparison (5-Fold Cross-Validation)
Model	ROC-AUC (CV)	Precision (>50K)	Recall (>50K)	F1 (>50K)
Logistic Regression	0.9070 ± 0.0013	0.7326	0.6008	0.6601
Random Forest	0.9168 ± 0.0019	0.7751	0.6050	0.6795
🔍 Key Observations

Logistic Regression provides a strong and stable baseline with good interpretability.

Random Forest improves overall performance, especially ROC-AUC and F1 score, by modeling non-linear relationships.

Precision remains high for both models, while recall for >50K remains challenging due to class imbalance.

Performance differences are meaningful but not extreme, reinforcing the importance of model choice based on use case.

🎯 Threshold Tuning & Decision Trade-Offs

Rather than relying on the default probability threshold (0.5), the Logistic Regression model was threshold-tuned.

At a tuned threshold (~0.35):

Recall for >50K increased from ~0.60 to ~0.74

Precision decreased accordingly

This highlights that classification performance is a decision, not a fixed outcome.

Such tuning may be preferred when missing high-income individuals is more costly than allowing additional false positives.

⚠️ Challenges Encountered

Several real-world ML challenges were intentionally addressed:

Handling string-based target labels during cross-validation

Correctly specifying the positive class for precision/recall metrics

Avoiding data leakage in preprocessing

Reconciling high ROC-AUC with lower recall

Managing notebook execution order in Colab

Understanding why accuracy alone is misleading on imbalanced data

Each issue was resolved explicitly to reinforce correct ML practices.

🧠 Key Lessons Learned

Proper pipelines are essential to avoid data leakage

Cross-validation provides more reliable estimates than single splits

ROC-AUC measures ranking, not classification decisions

Threshold tuning enables business-driven optimization

Linear and non-linear models serve different purposes

Clear reasoning matters more than complex models

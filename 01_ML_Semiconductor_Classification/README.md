# Machine Learning for Semiconductor Manufacturing — Pass/Fail Classification

A complete end-to-end machine learning pipeline applied to **real-world High Volume Manufacturing (HVM) semiconductor test data**. The goal: predict whether a unit will pass or fail Stage 2 quality testing based on Stage 1 measurements — enabling early failure detection and reducing downstream manufacturing costs.

> 📄 Completed as a final project for the Data Science Certificate Program — Instituto Tecnológico de Costa Rica (2023)
> 👥 Authors: Roberto Bocan · Jeison Araya

---

## 🏭 Business Context

In semiconductor manufacturing, identifying failing units early in the production flow is critical. Each unit that reaches final test as a failure represents wasted cost, time, and capacity. This project applies supervised machine learning to Stage 1 test measurements to predict Stage 2 outcomes — directly supporting yield optimization and manufacturing efficiency.

The dataset originates from a real HVM manufacturing process. Confidential values were anonymized prior to use; features are labeled A–G.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | HVM semiconductor manufacturing process |
| Initial size | 48,604 instances |
| Final size (after cleaning) | ~44,196 instances |
| Features | 7 numerical measurements (A, B, C, D, E, F, G) |
| Target | Binary — `1` = Pass, `0` = Fail |
| Class balance | **89.8% Pass / 10.2% Fail** (imbalanced ~10:1) |

---

## 🔄 Pipeline Overview

```
Raw Data (CSV)
     │
     ▼
 Preprocessing
  ├── Remove NaN targets (9.07% of data)
  ├── Filter Test_K outlier test type
  └── MinMax normalization [0, 1]
     │
     ▼
Exploratory Data Analysis
  ├── Class balance analysis
  ├── Pairplots & scatter plots
  └── Correlation heatmap
     │
     ▼
Model Training & Evaluation
  ├── Decision Tree (hyperparameter tuning: max_depth, min_samples_leaf)
  ├── SVC variants (SVC, LinearSVC, SGDClassifier)
  ├── K-Nearest Neighbors (n_neighbors sweep)
  ├── Naive Bayes (Gaussian, Complement, Bernoulli)
  └── Random Forest (n_estimators, max_depth tuning)
     │
     ▼
Imbalance Strategy: Random Undersampling
  └── Re-evaluate top models on balanced dataset
```

---

## 📏 Evaluation Metrics

Given the 10:1 class imbalance, accuracy alone is misleading. The following metrics were used:

- **Balanced Accuracy** — accounts for class imbalance
- **Cohen's Kappa** — agreement beyond chance
- **Matthews Correlation Coefficient (MCC)** — robust metric for imbalanced binary classification
- **Precision / Recall / F1** — trade-off between false positives and false negatives
- **Confusion Matrix** — direct view of false negatives (missed failures)

---

## 🏆 Results Summary

| Model | Test Accuracy | F1 Score | Kappa | MCC | Notes |
|-------|:---:|:---:|:---:|:---:|-------|
| **Decision Tree** | **99.14%** | **0.9952** | **0.9530** | **0.9532** | ✅ Best overall |
| **Random Forest** | 98.87% | 0.9937 | 0.9367 | 0.9383 | ✅ Excellent generalization |
| KNN (n=2) | 95.70% | — | — | — | ⚠️ 361 false negatives |
| SVC | ~89% | — | — | — | ❌ Poor on large dataset |
| LinearSVC | ~89% | — | — | — | ❌ Similar to SVC |
| SGDClassifier | ~89% | — | — | — | ❌ No improvement |
| Gaussian NB | — | — | 0.17 | 0.17 | ❌ Poor fit |
| Complement NB | 57.45% | — | — | — | ❌ Fails to generalize |
| Bernoulli NB | 89.61% | — | — | — | ❌ Always predicts Pass |

**Winner: Decision Tree** with `max_depth=30`, `min_samples_leaf=1`, `class_weight={0:1, 1:10}`

Only **120 out of ~11,000** test units were misclassified.

---

## ⚖️ Handling Class Imbalance

Two strategies were applied and compared:

1. **Class weighting** — penalizes misclassification of the minority (Fail) class during training using `class_weight={0:1, 1:10}`
2. **Random Undersampling** — reduces the majority class to a 2:1 ratio using `imblearn.RandomUnderSampler`, then re-evaluates all models

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-orange?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

**Models:** DecisionTreeClassifier · SVC · LinearSVC · SGDClassifier · KNeighborsClassifier · GaussianNB · ComplementNB · BernoulliNB · RandomForestClassifier

---

## 📁 Repository Structure

```
Data_Science/
└── ML_Project/
    ├── Proyecto_ML.ipynb     # Full notebook — preprocessing, EDA, modeling, results
    └── README.md
```

---

## 🔑 Key Takeaways

- **Accuracy is misleading on imbalanced data** — a model that always predicts "Pass" gets 89.8% accuracy but zero utility. MCC and Kappa are essential.
- **Decision Trees and Random Forests outperform SVMs and Naive Bayes** on this structured manufacturing dataset
- **Class weighting is more effective than undersampling** for this dataset — undersampling discards too much real data at a 10:1 ratio
- **Feature importance analysis** reveals which Stage 1 measurements are most predictive of Stage 2 failures — actionable insight for process engineers

---

## 👤 Author

**Roberto Bocan** — Product Development Engineer at Intel, specializing in semiconductor validation, thermal characterization, and data engineering.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rabocans)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Rbocan)

# Binary Classification: Model Comparison and Tuning

Final project for the **Machine Learning** course, University of Messina (2025).
Team project by **Sepehr Mojdehdehbaher** and **Fatemeh Baghchei** (equal contribution).

The goal is to predict a binary target from a tabular dataset and compare classic and ensemble models end to end: exploration, cleaning, feature engineering, training, tuning and interpretation.

Everything is in one notebook: [`ml_classification_project.ipynb`](ml_classification_project.ipynb).

## Data

`final_project_dataset_complete.csv`: **9,000 rows**, 8 numerical features, 2 categorical features and a binary `target` (4,721 class 0 / 4,279 class 1, so fairly balanced). Two numerical features have missing values (400 and 500 rows).

## What the notebook does

| Step | Content |
|---|---|
| 1. Exploration | Distributions, skewness and kurtosis, box plots for outliers, correlation heatmap |
| 2. Preprocessing | IQR outlier check; median imputation for skewed features, mean for the rest; standard scaling; ordinal encoding (`category_1`) and one-hot encoding (`category_2`) |
| 3. EDA | Features vs. target, pair plots, Welch t-tests (numerical) and chi-square tests (categorical) |
| 4. Feature engineering | Interaction and ratio terms for correlated pairs, polynomial features for the most target-correlated features, binning (12 → 51 features) |
| 5. Modelling | 9 models on a stratified 80/20 split with 5-fold cross-validation: Logistic Regression, KNN, Decision Tree, SVM (RBF), Random Forest, Gradient Boosting, AdaBoost, XGBoost, Stacking |
| 6. Tuning | `GridSearchCV` (5-fold) for Logistic Regression, Random Forest, Gradient Boosting, XGBoost and AdaBoost; default vs. tuned comparison |
| 7. Interpretation | Feature importances, SHAP, confusion matrices, analysis of misclassified samples |

## Results (test set, 1,800 rows)

| Model (tuned) | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** (best F1) | 0.893 | 0.901 | 0.870 | 0.885 | 0.962 |
| Logistic Regression | 0.892 | 0.893 | 0.877 | 0.885 | 0.959 |
| AdaBoost | 0.890 | 0.902 | 0.862 | 0.882 | 0.960 |
| XGBoost | 0.889 | 0.900 | 0.863 | 0.881 | 0.962 |
| Random Forest | 0.888 | 0.901 | 0.860 | 0.880 | 0.961 |

- Tuning improved accuracy only slightly (at most about +0.6 percentage points): most of the performance comes from preprocessing and feature engineering.
- A simple Logistic Regression is almost as good as the ensembles on this data.
- The untuned SVM predicted only one class (accuracy 0.52) even though its ROC AUC was high, so its decision threshold or parameters needed tuning; it was not kept.
- The most important features for Gradient Boosting were `feature_2`, `feature_1` and their polynomial terms.

## Limitations

- Imputation, scaling and the choice of polynomial features were fitted on the full dataset before the train/test split. A cleaner version would put these steps inside a scikit-learn `Pipeline` so the test set stays fully unseen.
- The features are anonymised (`feature_1` … `feature_8`), so the results cannot be interpreted in domain terms.

## How to run

```bash
pip install pandas numpy scikit-learn xgboost shap lime matplotlib seaborn scipy jupyter
jupyter notebook ml_classification_project.ipynb
```

Keep `final_project_dataset_complete.csv` in the same folder as the notebook.

## Tech

Python 3.12, pandas, NumPy, scikit-learn, XGBoost, SHAP, SciPy, Matplotlib, Seaborn, Jupyter.

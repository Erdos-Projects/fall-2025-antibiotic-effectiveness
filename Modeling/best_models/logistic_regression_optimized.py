# Import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from re import X
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (make_scorer, confusion_matrix, f1_score, precision_score, recall_score, PrecisionRecallDisplay)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

"""Model setup"""

# build pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=312)),
    ('logreg', LogisticRegression(max_iter=2000))
])

random_state = 312
# 5 kfold cv for hyperparameter otimization
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# model parameters
param_grid = {
    'logreg__C': uniform(0.1, 10),
    'logreg__solver': ['liblinear', 'lbfgs']
}

# Custom scorer for False Negative Rate
def fnr(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fnr = fn / (fn + tp)
    return fnr

# Scoring dictionary
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1),
    'f1': make_scorer(f1_score, pos_label=1),
    'roc_auc': 'roc_auc',
    'fnr': make_scorer(fnr),
    'f1_macro': make_scorer(f1_score, average='macro', pos_label=1),
    'f1_weighted': make_scorer(f1_score, average='weighted', pos_label=1)
}

"""Run model with hyperparameter tuning"""
dir = 'Data/final_dataframes/'
antibiotics = ['Gentamicin', 'Trimethoprim_Sulfamethoxazole', 'Ciprofloxacin',
                'Ampicillin', 'Cefazolin','Nitrofurantoin','Piperacillin_Tazobactam',
                'Levofloxacin', 'Ceftriaxone']
metric_results = []

for i in antibiotics:
    print('\n')
    print(f"==== Antibiotic: {i} ====")
    train_data = pd.read_csv(f'{dir}{i}_train_data.csv')
    X_train = train_data.drop(i, axis=1)
    y_train_raw = train_data[i]
    # one hot incoding such that resistance cases are positive cases
    y_train = (y_train_raw==2.0).astype(int)

    grid = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        cv=kf,
        scoring=scoring,
        n_jobs=-1,
        n_iter=100,
        random_state=random_state,
        refit='f1_weighted' # f1 weighted is the metric used to refit the model
    )

    # Fit grid search
    grid.fit(X_train, y_train)

    # Print best parameters per each antibiotic
    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

    # Get cross-validation results for all metrics
    results = grid.cv_results_

    print("\nAverage CV Metrics for Best Model:")

    # Save the metrics in a csv file
    scores = {"Antibiotic" : i}
    for metric in scoring.keys():
        mean_score = results[f'mean_test_{metric}'][grid.best_index_]
        std_score = results[f'std_test_{metric}'][grid.best_index_]
        scores[metric] = (mean_score, std_score)
        print(f"{metric}: {mean_score:.3f} ± {std_score:.3f}")

    metric_results.append(scores)

    # Get Predicted Probabilities on Validation Data
    best_model = grid.best_estimator_

    # Create and save PR curve and calibration plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    PrecisionRecallDisplay.from_estimator(best_model, X_train, y_train, ax=ax[0])
    ax[0].set_title("Precision-Recall Curve")

    CalibrationDisplay.from_estimator(best_model, X_train, y_train, ax=ax[1])
    ax[1].set_title("Calibration Plot")

    plt.tight_layout()
    plotdir = 'Modeling/best_models/training_metric_results/'
    plt.savefig(os.path.join(plotdir,f'{i}_calibration_PR_curve_plots.png'), dpi=300)

# Save validation metrics as a csv file
results_df = pd.DataFrame(metric_results)
savedir = 'Modeling/best_models/'
results_df.to_csv(os.path.join(savedir, 'loreg_optimized_metric_scores.csv'), index=False)


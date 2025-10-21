import pandas as pd
import numpy as np
import os
from re import X
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, confusion_matrix

antibiotics = ['Gentamicin', 'Trimethoprim_Sulfamethoxazole', 'Ciprofloxacin',
                'Ampicillin', 'Cefazolin','Nitrofurantoin','Piperacillin_Tazobactam',
                'Levofloxacin', 'Ceftriaxone']

# 5 kfold cv
random_state = 312
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
kf_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)

# Pipeline with Logistic Regression model
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight = 'balanced', max_iter=1000))
])

# Define FNR scorer (since sklearn doesn’t have one built-in)
def fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0

# custom scorer
fnr_scorer = make_scorer(fnr, greater_is_better=False)

# scoring metrics
scoring = {
    'f1' : make_scorer(f1_score, pos_label=1),
    'f1_weighted': make_scorer(f1_score, average='weighted', pos_label=1),
    'fnr': make_scorer(fnr)
}

# model parameters
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__solver': ['liblinear', 'lbfgs']
}

results_summary = []

for i in antibiotics:
    print(f"Antibiotic: {i}")
    train_data = pd.read_csv(f'Data/final_dataframes/{i}_train_data.csv')
    X_train = train_data.drop(i, axis=1)
    y_train_raw = train_data[i]
    y_train = (y_train_raw==2.0).astype(int)

    # --- Logistic Regression Model hyperparameter search ---
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=kf_inner,
        scoring='f1_weighted',
        n_jobs=-1,
        refit=True
    )
    nested_results_lr = cross_validate(
        grid, X_train, y_train, 
        cv=kf, 
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True
    )

    # --- Dummy Baseline Model ---
    dummy = DummyClassifier(strategy='stratified', random_state=312)
    nested_results_dummy = cross_validate(
        dummy, X_train, y_train,
        cv=kf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True
    )

     # --- Aggregate Results ---
    summary = {'Antibiotic': i}

    for metric in scoring.keys():
        # Logistic Regression
        train_lr = np.mean(nested_results_lr['train_' + metric])
        test_lr  = np.mean(nested_results_lr['test_' + metric])
        gap_lr = train_lr - test_lr

        # Dummy
        train_dummy = np.mean(nested_results_dummy['train_' + metric])
        test_dummy  = np.mean(nested_results_dummy['test_' + metric])

        summary[f'LR Train {metric}'] = train_lr
        summary[f'LR Test {metric}']  = test_lr
        summary[f'Dummy Train {metric}'] = train_dummy
        summary[f'Dummy Test {metric}']  = test_dummy

        print(f"\nMetric: {metric}")
        print(f"  Logistic Regression - Train: {train_lr:.4f}, Test: {test_lr:.4f}")
        print(f"  Dummy Baseline       - Train: {train_dummy:.4f}, Test: {test_dummy:.4f}")

        # Over/underfitting diagnostic
        if gap_lr > 0.05:
            print("  ⚠️ Potential overfitting in LR.")
        elif gap_lr < -0.05:
            print("  ❗Unexpected underfitting (test > train).")
        else:
            print("  ✅ LR generalizes well.")

    results_summary.append(summary)

# Save all results
results_df = pd.DataFrame(results_summary)
savedir = '/Modeling/LogReg/'
results_df.to_csv(os.path.join(savedir,'logistic_regression_vs_dummy_nestedCV.csv'), index=False)

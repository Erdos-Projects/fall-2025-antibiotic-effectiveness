# %%
import pandas as pd
import numpy as np

# Importing the XGBoost classifier we shall use later.
import xgboost as xgb
from xgboost import XGBClassifier

# Introducing Synthetic Minority Over-sampling Technique (SMOTE), which handles imbalanced binary datasets in which 0 or 1 dominates. 
from imblearn.over_sampling import SMOTE


from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    make_scorer, balanced_accuracy_score
)


from scipy.stats import randint, uniform
from sklearn.metrics import fbeta_score



# %%
# Loading the names for entibiotics, and reading their files.
antibiotics = ['Gentamicin', 'Trimethoprim_Sulfamethoxazole', 'Ciprofloxacin',
                'Ampicillin', 'Cefazolin','Nitrofurantoin','Piperacillin_Tazobactam',
                'Levofloxacin', 'Ceftriaxone']

# Fix a random state for reproducability.
n_randstat = 312 

# %%
for anti in antibiotics:

    all_results = []      #Creating dataframe to store model measurement results later.


    # Load and prepare data
    train_df = pd.read_csv(
        f"Final_dataframe-20251015T154934Z-1-001\\Final_dataframe\\{anti}_train_data.csv"
    )

    X = train_df.drop(columns=[anti, 'Year', 'anon_id'])
    y = train_df[anti] - 1  # Convert {1,2} to {0,1}


    # Define base pipeline  
    # Note: for XGBoost, we do not need employ Standard Scaler.
    base_pipeline = Pipeline([
         ('smote', SMOTE(              # Introducing SMOTE to handle imbalanced data.
        sampling_strategy=1.0,     
        random_state=n_randstat,             
        )),
        ('xgb', XGBClassifier(         # Introducing XGBoost classifier.
            objective='binary:logistic',
            eval_metric='auc',
            random_state=n_randstat
        ))
    ])

    
    # Hyperparameter space for random parameter search.
    param_distributions = {
        'xgb__n_estimators': list(range(50,200,1)),
        'xgb__learning_rate': uniform(0.01,0.9)       
    }


    # Scoring setup (all using Fβ)
    scoring = {
        'accuracy': make_scorer(accuracy_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),         # Maximizing recall is equivalent to minimizing fall negative rate, as they always sum up to be 1.
        'precision': make_scorer(precision_score, zero_division=0),
        'f1': make_scorer(f1_score, average = 'weighted', zero_division=0),      # Using weighted F1 score for model evaluation.
        'balanced_accuracy': make_scorer(balanced_accuracy_score, zero_division = 0),
    
    }



    # Nested CV setup
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=n_randstat)    # Outer 5-fold loop for cross-validation.
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=n_randstat)    # Inner 3-fold loop for hyper-parameter tuning.

    outer_results = []
    outer_fold = 1

    for train_idx, test_idx in outer_cv.split(X, y):                         # Outer 5-fold loop for cross-validation.
        print(f"\n==============================")
        print(f"🔹 Outer Fold {outer_fold} — training inner search for {anti}")
        print(f"==============================")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # ---- Inner hyperparameter tuning ----
        search = RandomizedSearchCV(                      # Defining the randomized searching module.
            estimator=base_pipeline,
            param_distributions=param_distributions,
            n_iter= 10,     # Iteration time for randomized searching.
            scoring= 'f1',  # using weighted f1 for optimization
            cv=inner_cv,    # Using a inner 3-fold for hyper-parameter tuning.
            random_state=n_randstat
        )

        search.fit(X_train, y_train)

        print("Best inner parameters:", search.best_params_)
        print(f"Best inner score:", search.best_score_)



        # ---- Evaluate on outer fold ----
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        fnr = 1 - recall          # False negative rate is equal to (1-recall).

        outer_results.append({
            'fold': outer_fold,
            'best_params': search.best_params_,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'FNR': fnr
        })

        outer_fold += 1    
    

    # Aggregate final performance
    outer_df = pd.DataFrame(outer_results)
    print(f"\n\n===== Nested CV Summary for {anti}=====")
    print(outer_df[['fold', 'recall', 'f1', 'FNR']])
    print("Mean Accuracy:", outer_df['accuracy'].mean())
    print("Mean Recall:", outer_df['recall'].mean())                
    print(f"Mean F1:", outer_df['f1'].mean())
    print("Mean false negative rate:", outer_df['FNR'].mean())

    # Save per-antibiotic result CSV
    output_path = f"results_{anti}.csv"
    outer_df.to_csv(output_path, index=False)         
    print(f"Saved results to {output_path}")

    # Add to results list
    all_results.append(outer_df)


# Merge all antibiotics' results together and save into .csv.
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv("all_antibiotics_results.csv", index=False)   



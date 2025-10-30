#------------------
##more rigorous hyperparamter searching specifically for SVM based on previous results
##designed for comparison to logistic regression script to determine final best model
#------------------

#import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from re import X
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (make_scorer, confusion_matrix, 
                             f1_score, precision_score, 
                             brier_score_loss, precision_recall_curve, 
                             accuracy_score, recall_score, get_scorer,
                             auc, roc_curve)
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform




##set up 

#antibiotics
antibiotics = ['Gentamicin', 'Trimethoprim_Sulfamethoxazole', 'Ciprofloxacin',
                'Ampicillin', 'Cefazolin','Nitrofurantoin','Piperacillin_Tazobactam',
                'Levofloxacin', 'Ceftriaxone']

#random state
random_state = 312

#kfold 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

#build pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=random_state)),
    ('svc', LinearSVC(dual='auto',max_iter=50000,random_state=random_state)),
])

#model parameters
param_grid = [
    {'svc__C':loguniform(0.3,3),
     'svc__loss':['squared_hinge'],
     'svc__penalty':['l1','l2']}
]

#custom scorer for False Negative Rate
def fnr(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fnr = fn / (fn + tp)
    return fnr

#scoring dictionary
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1),
    'roc_auc': 'roc_auc',
    'fnr': make_scorer(fnr),
    'f1_macro': make_scorer(f1_score, average='macro', pos_label=1),
    'f1_weighted': make_scorer(f1_score, average='weighted', pos_label=1)
}

#directories needed
datadir = 'Data/final_dataframes/'
savedir = 'Modeling/best_models/svm_results/'

#empty lists to save results in
train_results, test_results = [], []


#Model Trainer Class and associated functions, adjusted for SVM
class ModelTrainer:
    def __init__(self, param_grid, scoring, pipe, kf, random_state=312):
        self.param_grid = param_grid
        self.scoring = scoring
        self.pipe = pipe
        self.kf = kf
        self.random_state = random_state
    
    def train(self, X_train, y_train):
        # randomized serach for best hyper-parameters
        search = RandomizedSearchCV(
            estimator=self.pipe,
            param_distributions=self.param_grid,
            cv=self.kf,
            scoring=self.scoring,
            n_jobs=-1,
            n_iter=100,
            random_state=self.random_state,
            refit='f1_weighted'
        )
        search.fit(X_train, y_train)
        
        return search
    
    ## using naive probabilities as shown in 
    # https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html
    # so we can compare pre and post calibrated SVC model 
    def svc_naive_calibration(self, best_model, X_test):
        svc_df = best_model.decision_function(X_test)
        cal_df = (svc_df - svc_df.min()) / (svc_df.max() - svc_df.min())
        proba_pos_class = np.clip(cal_df,0,1)
        proba_neg_class = 1-proba_pos_class
        y_proba_pre = np.c_[proba_neg_class, proba_pos_class]
        
        return y_proba_pre

    def calibrate(self, best_model, X_train, y_train, imbalance_threshold = 0.5):
        # calibrate the best model
        # if class imbalance exceeds threshold, downsample negatives
        counts = y_train.value_counts(normalize=True)
        imbalance_ratio = counts.max()

        X_bal, y_bal = X_train.copy(), y_train.copy()
        if imbalance_ratio > imbalance_threshold:
            print(f"Downsampling negatives for calibration (ratio={imbalance_ratio:.2f}) > {imbalance_threshold}")
            df_train = pd.concat([X_train, y_train], axis=1)
            majority_class = y_train.mode()[0]
            minority_class = 1 - majority_class
            
            df_majority = df_train[df_train[y_train.name] == majority_class]
            df_minority = df_train[df_train[y_train.name] == minority_class]
            
            df_majority_downsampled = resample(
                df_majority,
                replace=False,
                n_samples=len(df_minority),
                random_state=self.random_state
            )
            df_balanced = pd.concat([df_majority_downsampled, df_minority])
            X_bal, y_bal = df_balanced.drop(y_train.name, axis=1), df_balanced[y_train.name]
        else:
            print(f"No downsampling applied (ratio={imbalance_ratio:.2f} > {imbalance_threshold})")

        #X_res, y_res = best_model.named_steps['smote'].fit_resample(X_train, y_train)
        calibrated_model = CalibratedClassifierCV(best_model.named_steps['svc'], method='sigmoid', cv=self.kf)
        calibrated_model.fit(X_bal, y_bal)

        return calibrated_model
    
##Model Evaluator Class and associated functions, adjusted for SVM
class ModelEvaluator:
    def __init__(self, scoring, output_dir):
        self.scoring = scoring
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_cv(self, search, antibiotic):
        # evaluate CV training dataset metrics
        cv_results = search.cv_results_
        scores = {"Antibiotic" : antibiotic}
        print("\nAverage CV Metrics for Best Model:")
        for metric in self.scoring.keys():
            mean_score = cv_results[f'mean_test_{metric}'][search.best_index_]
            scores[metric] = mean_score
            print(f"{metric}: {mean_score:.3f}")
        return scores
    
    def evaluate_test(self, best_model, svc_y_proba_pre, calibrated_model, X_test, y_test, antibiotic):
        # evaluate testing dataset metrics
        y_pred_pre= best_model.predict(X_test)

        #use self-defined function for 
        y_proba_pre = svc_y_proba_pre
        y_pred_post= calibrated_model.predict(X_test)
        y_proba_post = calibrated_model.predict_proba(X_test)[:, 1]

        test_result = {"Antibiotic" : antibiotic}
        test_result["brier_pre"] = brier_score_loss(y_test,y_proba_pre)
        test_result["brier_post"] = brier_score_loss(y_test, y_proba_post)

        for metric_name, scorer_key in self.scoring.items():
            scorer = get_scorer(scorer_key)
            test_result[f"{metric_name}_pre"] = scorer(best_model, X_test, y_test)
            test_result[f"{metric_name}_post"] = scorer(calibrated_model, X_test, y_test)
            print(f"{metric_name}_pre: {test_result[f'{metric_name}_pre']:.3f}")
            print(f"{metric_name}_post: {test_result[f'{metric_name}_post']:.3f}")

        print(f"brier_pre: {test_result['brier_pre']:.3f}")
        print(f"brier_post: {test_result['brier_post']:.3f}")
        print("\n")
        return test_result
    
#Plot Generator Class and associated functions
class PlotGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_all(self, antibiotic, y_test, y_pred_post, y_proba_post, y_proba_pre):
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Linear Support Vector Classifier Model Evaluation for {antibiotic}", fontsize=16, fontweight='bold')

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba_post)
        roc_auc = auc(fpr, tpr)
        ax[0,0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax[0,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax[0,0].set_title("ROC curve")
        ax[0,0].set_xlabel("False Positive Rate")
        ax[0,0].set_ylabel("True Positive Rate")
        ax[0,0].legend()

        # Precision Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba_post)
        ax[0,1].plot(recall, precision, color='purple', lw=2)
        ax[0,1].set_title("Precision-Recall Curve")
        ax[0,1].set_xlabel("Recall")
        ax[0,1].set_ylabel("Precision")

        # Calibration plot
        prob_true_pre, prob_pred_pre = calibration_curve(y_test, y_proba_pre, n_bins=10,strategy='quantile')
        prob_true_post, prob_pred_post = calibration_curve(y_test, y_proba_post, n_bins=10,strategy='quantile')
        ax[1,0].plot(prob_pred_pre, prob_true_pre, marker='x', linestyle='-', label="Before Calibration")
        ax[1,0].plot(prob_pred_post, prob_true_post, marker='o', linestyle='-', label="After Calibration")
        ax[1,0].plot([0,1],[0,1], color='navy', linestyle='--', label="Perfect Calibration")
        ax[1,0].set_title("Calibration Curve")
        ax[1,0].set_xlabel("Predicted Probability")
        ax[1,0].set_ylabel("True Probability")
        ax[1,0].legend()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_post)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1, 1])
        ax[1, 1].set_title("Confusion Matrix")
        ax[1, 1].set_xlabel("Predicted")
        ax[1, 1].set_ylabel("Actual")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, f"svm_modeleval_{antibiotic}.png"), dpi=300)
        plt.close()

    def plot_feature_importance(self, best_model, X_train, antibiotic):
        svc = best_model.named_steps['svc']
        feature_names = X_train.columns
        coefs = np.abs(svc.coef_.flatten())
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': coefs
        }).sort_values(by='importance', key=abs, ascending=False)

        # Save as CSV
        feature_importance.to_csv(os.path.join(self.output_dir, f"feature_importance_{antibiotic}.csv"), index=False)

        # Plot
        plt.figure(figsize=(8, 6))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature', palette='coolwarm')
        plt.title(f"Top Feature Importances - {antibiotic}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"feature_importance_{antibiotic}.png"))
        plt.close()

for i in antibiotics:
    print('\n')
    print(f"==== Antibiotic: {i} ====")
    train_data = pd.read_csv(f'{datadir}{i}_train_data.csv').drop(['Year', 'anon_id'], axis=1)
    test_data = pd.read_csv(f'{datadir}{i}_test_data.csv').drop(['Year', 'anon_id'], axis=1)

    X_train, y_train = train_data.drop(i, axis=1), (train_data[i]==2.0).astype(int)
    X_test, y_test = test_data.drop(i, axis=1), (test_data[i]==2.0).astype(int)

    #set up appropriate functions
    trainer = ModelTrainer(param_grid, scoring, pipe, kf, random_state)
    evaluator = ModelEvaluator(scoring, savedir)
    plotter = PlotGenerator(savedir)

    #identify best model
    search = trainer.train(X_train, y_train)
    best_model = search.best_estimator_

    #evaluate CV results
    train_results.append(evaluator.evaluate_cv(search,i))

    #get y_proba_pre scores for SVC
    svc_y_proba_pre = trainer.svc_naive_calibration(best_model, X_test)[:,1]

    #calibrate model 
    calibrated_model = trainer.calibrate(best_model, X_train, y_train, imbalance_threshold = 0.6)

    #evaluate model on the held out test set
    test_results.append(evaluator.evaluate_test(best_model, svc_y_proba_pre, calibrated_model, X_test, y_test, i))

    y_pred_pre = best_model.predict(X_test)
    y_pred_post = calibrated_model.predict(X_test)
    y_proba_post = calibrated_model.predict_proba(X_test)[:, 1]

    plotter.plot_all(i, y_test, y_pred_post, y_proba_post, svc_y_proba_pre)
    plotter.plot_feature_importance(best_model, X_train, antibiotic=i)

pd.DataFrame(test_results).to_csv(os.path.join(savedir, "svm_metrics_test_results.csv"), index=False)
pd.DataFrame(train_results).to_csv(os.path.join(savedir, "svm_metrics_train_results.csv"), index=False)



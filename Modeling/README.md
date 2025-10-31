**Folder Structure**

There are two folders, best_models and model_selection.

Model_selection covers scripts created to investigate each of the models: LogisticRegression, SVM, KNN, RandomForest and XGBoost, which each have their own folder documenting the initial investigation and experimentation done by group members for the models. In addition to that, we have the hyperparameteroptimizationallmodels.py script, which is a comprehensive hyperparameter searching script that covers all models based on the initial investigative work done by group members.

The results of hyperparameteroptimizationallmodels.py is contained in the "pkl_files" folder and "best_fit_plots_all_models" folder. The pkl_files folder contains metric results from the hyperparameter searching saved as pkl files, since the hyperparameter searhching takes approximately 2 hours to run. The pkl files can be loaded to plot results without re-running the hyperparameter search. The plots produced by the script of our hyperparameter searching are containing in the best_fit_plots_all_models. 

In the best_models folder, we have more rigorous hyperparameter searching logistic regression and support vector machines, which were chosen as the best models based on previous hyperparameter searching because they were better at minimizing false negative rates. Each model has its own script (svm_optimized.py and logistic_regression_optimized.py) that covers more rigorous parameter searching, and then comparing the results from the training and testing sets. The plots and csv files from these scripts are contained in the logreg_results and svm_results folders. 


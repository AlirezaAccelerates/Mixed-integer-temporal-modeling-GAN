import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB,MultinomialNB
from Choquet import ChoquetIntegral
from sklearn.metrics import roc_auc_score

# Set random seeds
np.random.seed(2023)
random_state = 2023

# Load the data
data_fluid_overload_imputed = pd.read_csv("./data_fluid.overload_imputed.csv")

# Calculate the number of imputed datasets
total_datasets = len(data_fluid_overload_imputed) // 991

# Split the DataFrame into individual datasets
datasets = [data_fluid_overload_imputed.iloc[i*991:(i+1)*991] for i in range(total_datasets)]

# Randomly select 10 datasets
selected_datasets = np.random.choice(datasets, size=10, replace=False)

# Combine the selected datasets into a single DataFrame
data_fluid_overload = pd.concat(selected_datasets)

# Data processing
data = data_fluid_overload.copy()
sex_cols = pd.get_dummies(data['SEX'])
ICU_type_cols = pd.get_dummies(data['main_ICU_type'])
label = data['fluid_overload']
data= data.drop(columns = ['SEX','main_ICU_type', 'fluid_overload'],axis=1)
data = pd.concat([sex_cols['Male'], ICU_type_cols['Medical'],data], axis=1)
data['AKI_24h'] = data['AKI_24h'].map({'Yes': 1, 'No': 0})
data['CRRT_24h'] = data['CRRT_24h'].map({'Yes': 1, 'No': 0})
data['CRRT'] = data['CRRT'].map({'Yes': 1, 'No': 0})
data['Mechanical_ventilation'] = data['Mechanical_ventilation'].map({'Yes': 1, 'No': 0})
data['Vasopressor_24h'] = data['Vasopressor_24h'].map({'Yes': 1, 'No': 0})
data['Use_CI_24h'] = data['Use_CI_24h'].map({'Yes': 1, 'No': 0})

# Define the classifiers' parameter grids
rf_grid = {'n_estimators': [100, 150, 200, 300, 500, 1000, 1500, 3000], 'max_depth': [5, 8, 10, 12, 14, 18], 
                      'min_samples_split': [1, 2, 4], 'min_samples_split': [2, 4, 5, 10], 'class_weight': [{0.0:1,1.0:1},{0.0:1,1.0:1.2},{0.0:1,1.0:1.5}], 'random_state': [random_state]}
svm_grid = {'C': [0.2, 0.5, 0.8, 1.5, 3, 5, 10, 25, 50], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3, 4, 8], 'random_state': [random_state]}
xgb_grid = {'n_estimators': [100, 250, 500], 'max_depth': [5, 7, 12, 15], 'learning_rate': [0.01, 0.1], 'colsample_bytree': [0.6, 0.8, 1],
                'gamma': [0, 0.1, 1], 'scale_pos_weight': [1, 1.1, 1.2, 1.5, 2], 'random_state': [random_state]}

# Search space for the meta-model
dt_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
ada_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}

meta_grids = {
    'LogisticRegression': None,
    'DecisionTree': dt_grid,
    'AdaBoost': ada_grid,
    'CustomModel': None  # 'None' indicates no hyperparameters for 'ch'
}

# Stratified K-Folds cross-validator
cv = StratifiedKFold(n_splits=5)
cv2 = StratifiedKFold(n_splits=5)

# Initialize the best score and best params
best_score = 0
best_params = {}

# Iterate over all combinations of hyperparameters
for rf_params in ParameterGrid(rf_grid):
    for svm_params in ParameterGrid(svm_grid):
        for xgb_params in ParameterGrid(xgb_grid):
            ensemble_scores = []

            for train_idx, test_idx in cv.split(data, label):
                X_train, X_test = data[train_idx], data[test_idx]
                y_train, y_test = label[train_idx], label[test_idx]

                for opt_idx, val_idx in cv2.split(X_train, y_train):
                    X_opt, X_val = X_train[opt_idx], X_train[val_idx]
                    y_opt, y_val = y_train[opt_idx], y_train[val_idx]  
                    ensemble_scores_inner = []

                    # Train each base model with the current set of parameters
                    rf_model = RandomForestClassifier(**rf_params).fit(X_opt, y_opt)
                    svm_model = SVC(**svm_params, probability=True).fit(X_opt, y_opt)
                    xgb_model = xgb.XGBClassifier(**xgb_params).fit(X_opt, y_opt)

                    # Generate predictions (probabilities) from each model
                    rf_pred = rf_model.predict_proba(X_val)
                    svm_pred = svm_model.predict_proba(X_val)
                    xgb_pred = xgb_model.predict_proba(X_val)

                    # Combine the predictions for the meta-model
                    meta_features = np.column_stack([rf_pred, svm_pred, xgb_pred])

                    # Train the meta-model
                    meta_model = GaussianNB().fit(meta_features, y_val)

                    # Evaluate the meta-model
                    meta_model_pred = meta_model.predict(meta_features)
                    accuracy = roc_auc_score(y_val, meta_model_pred)

                    ensemble_scores_inner.append(accuracy)

                # Average score of the inner loop
                avg = np.append(np.mean(ensemble_scores_inner))

            # Average score for this combination
            avg_score = np.mean(avg)

            # Update best score and parameters if current score is better
            if avg_score > best_score:
                best_score = avg_score
                best_params = {'rf': rf_params, 'svm': svm_params, 'xgb': xgb_params}

# Print the best score and corresponding parameters
print("Best Ensemble Model Accuracy:", best_score)
print("Best Parameters:", best_params)


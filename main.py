import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from imblearn.over_sampling import SMOTE

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
data= data.drop(columns = ['SEX','main_ICU_type'],axis=1)
data = pd.concat([sex_cols['Male'], ICU_type_cols['Medical'],data], axis=1)
data['AKI_24h'] = data['AKI_24h'].map({'Yes': 1, 'No': 0})
data['CRRT_24h'] = data['CRRT_24h'].map({'Yes': 1, 'No': 0})
data['CRRT'] = data['CRRT'].map({'Yes': 1, 'No': 0})
data['Mechanical_ventilation'] = data['Mechanical_ventilation'].map({'Yes': 1, 'No': 0})
data['Vasopressor_24h'] = data['Vasopressor_24h'].map({'Yes': 1, 'No': 0})
data['Use_CI_24h'] = data['Use_CI_24h'].map({'Yes': 1, 'No': 0})

# Define the classifiers and their parameter grids
classifiers = {
    'SVM': [SVC(), {'C': [0.2, 0.5, 0.8, 1.5, 3, 5, 10, 25, 50], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3, 4, 8], 'random_state': [random_state]}],
    'Random Forest': [RandomForestClassifier(), {'n_estimators': [100, 150, 200, 300, 500, 1000, 1500, 3000], 'max_depth': [5, 8, 10, 12, 14, 18], 
                      'min_samples_split': [1, 2, 4], 'min_samples_split': [2, 4, 5, 10], 'class_weight': [{0.0:1,1.0:1},{0.0:1,1.0:1.2},{0.0:1,1.0:1.5}], 'random_state': [random_state]}],
    'XGBoost': [xgb.XGBClassifier, {'n_estimators': [100, 250, 500], 'max_depth': [5, 7, 12, 15], 'learning_rate': [0.01, 0.1], 'colsample_bytree': [0.6, 0.8, 1],
                'gamma': [0, 0.1, 1], 'scale_pos_weight': [1, 1.1, 1.2, 1.5, 2], 'random_state': [random_state]}]
}

rf_grid = {'n_estimators': [100, 150, 200, 300, 500, 1000, 1500, 3000], 'max_depth': [5, 8, 10, 12, 14, 18], 
                      'min_samples_split': [1, 2, 4], 'min_samples_split': [2, 4, 5, 10], 'class_weight': [{0.0:1,1.0:1},{0.0:1,1.0:1.2},{0.0:1,1.0:1.5}], 'random_state': [random_state]}
svm_grid = {'C': [0.2, 0.5, 0.8, 1.5, 3, 5, 10, 25, 50], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3, 4, 8], 'random_state': [random_state]}
xgb_grid = {'n_estimators': [100, 250, 500], 'max_depth': [5, 7, 12, 15], 'learning_rate': [0.01, 0.1], 'colsample_bytree': [0.6, 0.8, 1],
                'gamma': [0, 0.1, 1], 'scale_pos_weight': [1, 1.1, 1.2, 1.5, 2], 'random_state': [random_state]}


# Stratified K-Folds cross-validator
cv = StratifiedKFold(n_splits=5)

# Iterate over classifiers and hyperparameters
best_params = {}
for name, (classifier, param_grid) in classifiers.items():
    best_score = 0
    for params in ParameterGrid(param_grid):
        classifier.set_params(**params)
        scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Apply SMOTE
            smote = SMOTE()
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            # Train the model
            classifier.fit(X_train_smote, y_train_smote)
            y_pred = classifier.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params[name] = params

# Print the best parameters for each classifier
for classifier in best_params:
    print(f"Best parameters for {classifier}: {best_params[classifier]}")
    print(f"Best cross-validation score for {classifier}: {best_score}")
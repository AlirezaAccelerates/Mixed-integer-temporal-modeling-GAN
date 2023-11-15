import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Search space
param_grids = {
    'SVM': {'classifier__C': [0.2, 0.5, 0.8, 1.5, 3, 5, 10, 25, 50], 'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'classifier__degree': [2, 3, 4, 8], 'classifier__random_state': [random_state]},
    'Random Forest': {'classifier__n_estimators': [100, 150, 200, 300, 500, 1000, 1500, 3000], 'classifier__max_depth': [5, 8, 10, 12, 14, 18], 
                      'classifier__min_samples_split': [1, 2, 4], 'classifier__min_samples_split': [2, 4, 5, 10], 'classifier__class_weight': [{0.0:1,1.0:1},{0.0:1,1.0:1.2},{0.0:1,1.0:1.5}], 'classifier__random_state': [random_state]},
    'XGBoost': {'classifier__n_estimators': [100, 250, 500], 'classifier__max_depth': [5, 7, 12, 15], 'classifier__learning_rate': [0.01, 0.1], 'classifier__colsample_bytree': [0.6, 0.8, 1],
                'classifier__gamma': [0, 0.1, 1], 'classifier__scale_pos_weight': [1, 1.1, 1.2, 1.5, 2], 'classifier__random_state': [random_state]}
}

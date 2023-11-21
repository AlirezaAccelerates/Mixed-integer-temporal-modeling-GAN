# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB, MultinomialNB
from Choquet import ChoquetIntegral
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

# Set random seeds
np.random.seed(2023)
random_state = 2023

# Load the data
data_fluid_overload_imputed = pd.read_csv("./data_fluid.overload_imputed.csv")
unique_pat_num = 991

# Calculate the number of imputed datasets
total_datasets = len(data_fluid_overload_imputed) // unique_pat_num

# Split the DataFrame into individual datasets
datasets = [data_fluid_overload_imputed.iloc[i*unique_pat_num:(i+1)*unique_pat_num] for i in range(total_datasets)]

# Randomly select ten subdatasets
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
mlp_grid = {'hidden_layer_sizes': [(8,32,8), (16,32,8), (32,16,8), (32,64,16)],'random_state': [random_state]}
nb_grid = {'var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7], 'random_state': [random_state]}
non_grid = {'random_state': [random_state]}
meta_grids = {
    'RandomForest': rf_grid,
    'MLP': mlp_grid,
    'GNaiveBayes': nb_grid,
    'BNaiveBayes': non_grid,
    'CNaiveBayes': non_grid,
    'MNaiveBayes': non_grid,
    'LogisticRegression': non_grid,
    'Voting': None,
    'Choquet': None
}

# Stratified K-Folds cross-validator
cv = StratifiedKFold(n_splits=5)
cv2 = StratifiedKFold(n_splits=5)

# Iterate over all imputed dataset and combinations of hyperparameters
for ite in range(10):
    # Initialize the best score and best params
    best_score = 0
    best_score_meta = 0
    best_params = {}
    data_fold = data.iloc[991*ite:991*(ite+1),:]
    label_fold = label.iloc[991*ite:991*(ite+1),:]
    for rf_params in ParameterGrid(rf_grid):
        for svm_params in ParameterGrid(svm_grid):
            for xgb_params in ParameterGrid(xgb_grid):
                # Find the best meta-model
                for meta_name, meta_grid in meta_grids.items():
                    for meta_params in ParameterGrid(meta_grid):
                        ensemble_scores = []
                        # Instantiate meta-model based on the type
                        if meta_name == 'RandomForest':
                            meta_model = RandomForestClassifier(**meta_params)
                        elif meta_name == 'MLP':
                            meta_model = MLPClassifier(**meta_params)
                        elif meta_name == 'GNaiveBayes':
                            meta_model = GaussianNB(**meta_params)
                        elif meta_name == 'BNaiveBayes':
                            meta_model = BernoulliNB(**meta_params)
                        elif meta_name == 'CNaiveBayes':
                            meta_model = ComplementNB(**meta_params)
                        elif meta_name == 'MNaiveBayes':
                            meta_model = MultinomialNB(**meta_params)
                        elif meta_name == 'Voting':
                            meta_model = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model), ('xgb', xgb_model)], voting='soft')  
                        elif meta_name == 'Choquet':
                            meta_model = ChoquetIntegral()                    

                        for train_idx, test_idx in cv.split(data_fold, label_fold):
                            X_train, X_test = data_fold[train_idx], data_fold[test_idx]
                            y_train, y_test = label_fold[train_idx], label_fold[test_idx]

                            for opt_idx, val_idx in cv2.split(X_train, y_train):
                                X_opt, X_val = X_train[opt_idx], X_train[val_idx]
                                y_opt, y_val = y_train[opt_idx], y_train[val_idx]  
                                ensemble_scores_inner = []

                                # Apply SMOTE
                                smote = SMOTE(random_state = random_state)
                                X_opt_smote, y_opt_smote = smote.fit_resample(X_opt, y_opt)
                                opt_data = pd.concat([X_opt, y_opt], axis=1)

                                # Create and train the CTGAN model. To have more control over
                                # the ctgan synthetic data generator, SDV library should be used.
                                # We write this part here using ctgan library, which is a part of 
                                # the SDV library for simplicit. Number of epochs is a hyperparameter here.
                                ctgan = CTGAN()
                                ctgan.fit(opt_data.astype('float'), epochs=epochs)
                                # Generate synthetic data with the CTGAN model
                                num_opt_data_ctgan = len(opt_data)
                                opt_data_ctgan = ctgan.sample(num_opt_data_ctgan)
                                X_opt_ctgan = opt_data_ctgan.iloc[:,:-1]
                                y_opt_ctgan = opt_data_ctgan.iloc[:,-1]
                                # Prepare the training data
                                X_opt = pd.concat([X_opt, X_opt_smote, X_opt_ctgan], axis =1)
                                y_opt = pd.concat([y_opt, y_opt_smote, y_opt_ctgan], axis =1)

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
                                meta_model = meta_model.fit(meta_features, y_val)

                                # Evaluate the meta-model
                                meta_model_pred = meta_model.predict(meta_features)
                                auc = roc_auc_score(y_val, meta_model_pred)

                                ensemble_scores_inner.append(auc)

                        # Average score of the inner loop
                        avg = np.append(np.mean(ensemble_scores_inner))

                    # Average score for this combination
                    avg_score = np.mean(avg)

                    # Update best score and parameters if current score is better
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {'rf': rf_params, 'svm': svm_params, 'xgb': xgb_params,
                                    'meta': meta_params, 'meta_model': meta_name}

    # Print the best parameters for every iteration
    print(f"Best Parameters for iteration {ite} is:", best_params)
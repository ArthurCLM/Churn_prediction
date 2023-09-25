# DATA MANIPULATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# MODELLING
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.base import clone

# WARNINGS.
import warnings
warnings.filterwarnings('ignore')

# UTILS
from modelling_functions import *


df = pd.read_csv("/Data/Churn_Modelling.csv")

# SEPARATING TRAIN AND TEST DATASETS
X = df.drop(columns=['Exited'])
y = df['Exited'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# PREPROCESSING THE DATA
X_train_prepared = DfPrepPipeline(X_train)
X_test_prepared = DfPrepPipeline(X_test)

# TUNNING MODEL
n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

search_space = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(2,16),
    'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'min_child_weight': Integer(1,10),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'lambda': Real(0.0, 10.0),
    'gamma': Real(0.0, 10.0),
    'scale_pos_weight': Integer(1, 10)
}

bayesian_search = BayesSearchCV(XGBClassifier(), search_space, cv=stratified_kfold, n_iter=50, 
                                scoring='roc_auc', return_train_score=True, random_state=1) 

print(f'The best params found for XGBoost are: ')
print(bayesian_search.best_params_)

# EVALUATE ALL DATA
X_all_data = pd.concat([X_train_prepared, X_test_prepared], axis=0)
y_all_data = pd.concat([y_train, y_test], axis=0)

final_predictions = final_model_par.predict(X_all_data)
probas = final_model_par.predict_proba(X_all_data)[:, 1]

evaluate_classifier(y_all_data, final_predictions, probas)

print("END.")









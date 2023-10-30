# Data Processing
import pandas as pd
import numpy as np
import sklearn.metrics
import torch
# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from sklearn.metrics import f1_score, make_scorer
import random
from sklearn.metrics import roc_auc_score, f1_score
from skopt import BayesSearchCV
from optuna.samplers import TPESampler
import lightgbm as lgb
from lightgbm import log_evaluation
import optuna.integration.lightgbm as lgbm
import optuna

# optuna.logging.set_verbosity(optuna.logging.WARNING)
scaler = StandardScaler()
device = 'cuda'

def preprocessing(df):
    # Columns with missing value: AdmitDiagnosis, religion, marital_status, LOSgroupNum
    # Dropping marital_status, LOSgroupNum because too many missing values; AdmitDiagnosis, AdmitProcedure because too many classes.
    df.drop(labels=['LOSgroupNum'], axis=1, inplace=True)
    categorical_features = ['gender', 'admit_type', 'admit_location', 'insurance', 'religion', 'ethnicity', 'AdmitDiagnosis', 'AdmitProcedure', 'marital_status']
    y = pd.DataFrame(df.pop('ExpiredHospital'))
    x1 = df.drop(labels=categorical_features, axis=1)
    x2 = df[categorical_features]
    x2 = x2.apply(lambda x: pd.factorize(x)[0])
    data = pd.concat([x1, x2], axis=1)
    return data, y

def objective(trial):
    """
    Objective function to be minimized.
    """
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)
    x_train, y_train = preprocessing(train_df)
    x_val, y_val = preprocessing(val_df)
    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val)

    params = {
        # 'device': 'gpu',
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_class": 1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),  # Boosting learning rate
        # 'n_estimators': trial.suggest_int('n_estimators', 550, 750),  # Number of boosted trees to fit
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 100, 400),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    # 600: 8.10 550: 8.14
    gbm = lgb.train(params,
                    dtrain,
                    num_boost_round=1000,
                    valid_sets=[dtrain, dval],
                    callbacks=[lgb.early_stopping(100),log_evaluation(100)])
    y_pred = gbm.predict(x_val)
    y_pred = np.rint(y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    # accuracy = accuracy_score(y_val, y_pred)
    # roc_auc = roc_auc_score(y_val, y_pred)
    return f1_macro


def train():
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)
    x_train, y_train = preprocessing(train_df)
    x_val, y_val = preprocessing(val_df)
    dtrain = lgb.Dataset(x_train, label=y_train)

    study = optuna.create_study(study_name="lightgbm", direction="minimize")
    study.optimize(objective, n_trials=50)
    best_model = lgb.LGBMClassifier(**study.best_params)
    best_model.fit(x_train, y_train.values.ravel())
    y_pred = best_model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f'F1: {f1} | Roc_auc: {roc_auc} | Recall: {recall} | Accuracy: {accuracy}')
    return f1, accuracy, best_model

def predict(df):
    x, _ = preprocessing(df)
    loaded_rf = joblib.load('lgbm_classifier.joblib')
    y_pred = loaded_rf.predict(x)
    pd.DataFrame(y_pred).to_csv('lgbm-preds.csv')
    print('Saved!')


if __name__ == '__main__':
    max_f1 = 0.67
    max_acc = 0.94
    for i in range(10):
        f1, acc, model = train()
        if f1 > max_f1 and acc > max_acc:
            max_f1 = f1
            max_acc = acc
            joblib.dump(model, 'lgbm_classifier.joblib')
            print("Saved: ", end="")
            print(f'New max f1: {max_f1}')
            print(model.get_params())
    # predict(pd.read_csv('Assignment3-Unknown-Dataset.csv'))










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
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
scaler = StandardScaler()
device = 'cuda'

def preprocessing(df):

    df.drop(labels=['LOSgroupNum'], axis=1, inplace=True)
    categorical_features = ['gender', 'admit_type', 'admit_location', 'insurance', 'religion', 'ethnicity', 'AdmitDiagnosis', 'AdmitProcedure', 'marital_status']
    y = pd.DataFrame(df.pop('ExpiredHospital'))
    x1 = df.drop(labels=categorical_features, axis=1)
    x2 = df[categorical_features]
    x2 = x2.apply(lambda x: pd.factorize(x)[0])
    data = pd.concat([x1, x2], axis=1)
    return data, y


def report_perf(optimizer, x_train, y_train, title="model"):

    optimizer.fit(x_train, y_train.values.ravel())

    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    print((title + "Candidates checked: %d, best CV score: %.3f "
           + u"\u00B1" + " %.3f") % (len(optimizer.cv_results_['params']),
                                     best_score,
                                     best_score_std))
    print('Best parameters:', best_params)
    return best_params


def train(df):
    categorical_features = ['gender', 'admit_type', 'admit_location', 'insurance', 'religion', 'ethnicity',
                            'AdmitDiagnosis', 'AdmitProcedure', 'marital_status']
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)
    x_train, y_train = preprocessing(train_df)
    x_val, y_val = preprocessing(val_df)
    scale = df['ExpiredHospital'].value_counts()[0]/df['ExpiredHospital'].value_counts()[1]

    fixed_params = {
        'n_jobs': 2,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'f1_macro',
        'scale_pos_weight': scale,
        'bagging_freq': 1,
        'learning_rate': 0.03,
        'n_estimators': 450,
    }

    model = lgb.LGBMClassifier(**fixed_params)
    search_spaces = {
        # 'learning_rate': Real(0.02, 0.03, 'log-uniform'),
        # 'n_estimators': Integer(300, 500),
        'num_leaves': Integer(200, 300),
        'max_depth': Integer(15, 30),
        'bagging_fraction': (0.7, 0.9),
        # 'reg_alpha': Real(0.3, 0.7, 'log-uniform'),  # L1 regularization
        # 'reg_lambda': Real(0.2, 0.5, 'log-uniform'),      # L2 regularization
    }

    fit_params = {
        'early_stopping_rounds': 200,
        'eval_set': [(x_val, y_val)],
        'eval_metric': 'f1_macro'
    }

    opt = BayesSearchCV(estimator=model,
                        search_spaces=search_spaces,
                        scoring='f1_macro',
                        cv=5,
                        n_iter=5,  # max number of trials
                        n_points=3,  # number of hyperparameter sets evaluated at the same time
                        n_jobs=2,  # number of jobs
                        return_train_score=True,
                        refit=False,
                        optimizer_kwargs={'base_estimator': 'GP'},  # optmizer parameters: we use Gaussian Process (GP)
                        # fit_params=fit_params,
                        verbose=0,
                        )
    best_params = report_perf(opt, x_train, y_train, 'LightGBM_classifier')
    best_model = lgb.LGBMClassifier(
                                    **best_params,
                                    **fixed_params,
                                    )

    best_model.fit(x_train, y_train.values.ravel())
    y_pred = best_model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred)
    f1_mac = f1_score(y_val, y_pred, average='macro')
    print(f'F1_mac: {f1_mac} | Roc_auc: {roc_auc} | Recall: {recall} | Accuracy: {accuracy}')
    return f1_mac, accuracy, best_model

def predict(df):
    x, _ = preprocessing(df)
    loaded_rf = joblib.load('lgbm_classifier.joblib')
    y_pred = loaded_rf.predict(x)
    pd.DataFrame(y_pred).to_csv('lgbm-preds.csv')
    print('Saved!')

if __name__ == '__main__':
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    max_f1_mac = 0.824
    max_acc = 0.94
    for i in range(100):
        print("Interation:", i)
        f1_mac, acc, model = train(df)
        if f1_mac > max_f1_mac:
            max_f1_mac = f1_mac
            max_acc = acc
            joblib.dump(model, 'lgbm_classifier.joblib')
            print("Saved: ", end="")
            print(f'New max f1 macro: {max_f1_mac}')
            print(model.get_params())
    predict(pd.read_csv('Assignment3-Unknown-Dataset.csv'))









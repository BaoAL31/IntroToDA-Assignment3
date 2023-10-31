# Data Processing
import pandas as pd
import numpy as np
import torch
# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import random
scaler = StandardScaler()
device = 'cuda'

def preprocessing(df):
    # Columns with missing value: AdmitDiagnosis, religion, marital_status, LOSgroupNum
    # Dropping marital_status, LOSgroupNum because too many missing values; AdmitDiagnosis, AdmitProcedure because too many classes.
    df.drop(labels=['marital_status', 'LOSgroupNum'], axis=1, inplace=True)

    categorical_features = ['gender', 'admit_type', 'admit_location', 'insurance', 'religion', 'ethnicity', 'AdmitDiagnosis', 'AdmitProcedure']
    y = pd.DataFrame(df.pop('ExpiredHospital'))
    x1 = df.drop(labels=categorical_features, axis=1)
    x1.fillna(np.floor(x1.mean()), inplace=True)
    # x1_scaled = (x1-x1.min())/(x1.max()-x1.min())
    x2 = df[categorical_features]
    x2 = x2.fillna(x2.mode().iloc[0])
    x2 = x2.apply(lambda x: pd.factorize(x)[0])
    data = pd.concat([x1, x2], axis=1)
    return data, y

def train(df):
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)
    x_train, y_train = preprocessing(train_df)
    x_val, y_val = preprocessing(val_df)
    scale = df['ExpiredHospital'].value_counts()[0]/df['ExpiredHospital'].value_counts()[1]

    model = XGBClassifier()
    fixed_params = {
        'n_jobs': 2,
        'objective': 'binary:logistic',
        'scale_pos_weight': scale,
        'subsample': 0.9,
        # 'learning_rate': 0.05,
    }
    model.set_params(**fixed_params)

    params_dist = {
        'n_estimators': randint(100, 400),
        'max_depth': randint(10, 30),
        'learning_rate': [0.02, 0.05, 0.08],
        # 'min_child_weight': randint(2, 5),
    }

    random_params = RandomizedSearchCV(model,
                                       scoring='f1_macro',
                                       param_distributions=params_dist,
                                       n_iter=100,
                                       cv=5,
                                       n_jobs=2,
                                       verbose=3,
                                       )
    random_params.fit(x_train.values, y_train.values.ravel())
    print('Best hyperparameters:', random_params.best_params_)
    # best_model = model
    best_model = random_params.best_estimator_
    best_model.fit(x_train, y_train.values.ravel())
    y_pred = best_model.predict(x_val)
    print(classification_report(y_val, y_pred))
    # joblib.dump(best_model, 'xgb_classifier.joblib')

def predict(df):
    x, _ = preprocessing(df)
    loaded_rf = joblib.load('xgb_classifier.joblib')
    y_pred = loaded_rf.predict(x)
    pd.DataFrame(y_pred).to_csv('xgb-preds.csv')
    print('Saved!')

if __name__ == '__main__':
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    train(df)
    # predict(pd.read_csv('Assignment3-Unknown-Dataset.csv'))









# Data Processing
import pandas as pd
import numpy as np
import sklearn.metrics
# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from sklearn.metrics import f1_score, make_scorer
import random
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb
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
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    x_train, y_train = preprocessing(train_df)
    x_val, y_val = preprocessing(val_df)

    fixed_params = {
        # 'learning_rate': 0.05,
        # 'n_estimators': 320,
        # 'num_leaves': 200,
        "max_depth": 22,
        "min_child_samples": 7,
        "colsample_bytree": 0.8738172709258141,
        "subsample": 0.7683619877865955,
        "min_split_gain": 0.5
    }

    model = lgb.LGBMClassifier( boosting_type='gbdt',
                                objective='binary',
                                metric='f1_weighted',
                                n_jobs=2,
                                verbose=-1,
                                is_unbalance=True,
                                # random_state=42,
                              )
    search_spaces = {
        'learning_rate': Real(0.05, 0.5, 'log-uniform'),  # Boosting learning rate
        'n_estimators': Integer(200, 600),  # Number of boosted trees to fit
        'num_leaves': Integer(100, 500),  # Maximum tree leaves for base learners
        'max_depth': Integer(10, 24),  # Maximum tree depth for base learners, <=0 means no limit

    }

    opt = BayesSearchCV(estimator=model,
                        search_spaces=search_spaces,
                        scoring='f1_weighted',
                        cv=5,
                        n_iter=5,  # max number of trials
                        n_points=2,  # number of hyperparameter sets evaluated at the same time
                        n_jobs=2,  # number of jobs
                        return_train_score=True,
                        refit=False,
                        optimizer_kwargs={'base_estimator': 'GP'},  # optmizer parameters: we use Gaussian Process (GP)
                        # random_state=42,
                        )

    best_params = report_perf(opt, x_train, y_train, 'LightGBM_classifier')
    best_model = lgb.LGBMClassifier(boosting_type='gbdt',
                                    objective='binary',
                                    metric='f1_weighted',
                                    n_jobs=1,
                                    verbose=-1,
                                    is_unbalance=True,
                                    # random_state=42,
                                    **best_params,
                                    )

    best_model.fit(x_train, y_train.values.ravel())
    y_pred = best_model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    f1_mac = f1_score(y_val, y_pred, average='macro')
    print(f'F1: {f1} | F1_mac: {f1_mac} | Recall: {recall} | Accuracy: {accuracy}')

    return f1, accuracy, model

def predict(df):
    x, _ = preprocessing(df)
    loaded_rf = joblib.load('lgbm_classifier.joblib')
    y_pred = loaded_rf.predict(x)
    pd.DataFrame(y_pred).to_csv('lgbm-preds.csv')
    print('Saved!')

if __name__ == '__main__':
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    max_f1 = 0.66
    max_acc = 0.935
    preprocessing(df)
    # for i in range(50):
    #     f1, acc, model = train(df)
    #     if f1 > max_f1 and acc > max_acc:
    #         max_f1 = f1
    #         max_acc = acc
    #         joblib.dump(model, 'lgbm_classifier.joblib')
    #         print("Saved: ", end="")
    #         print(f'New max f1: {max_f1}, acc: {max_acc}')
    #         print(model.get_params())
    # predict(pd.read_csv('Assignment3-Unknown-Dataset.csv'))









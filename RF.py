# Data Processing
import pandas as pd
import numpy as np
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import export_graphviz
from scipy.stats import randint
from skopt.space import Real, Categorical, Integer
import joblib


def preprocessing(df):
    # Columns with missing value: AdmitDiagnosis, religion, marital_status, LOSgroupNum
    # Dropping marital_status, LOSgroupNum because too many missing values; AdmitDiagnosis, AdmitProcedure because too many classes.
    df.drop(labels=['LOSgroupNum'], axis=1, inplace=True)

    categorical_features = ['gender', 'admit_type', 'admit_location', 'insurance', 'marital_status', 'religion', 'ethnicity', 'AdmitDiagnosis', 'AdmitProcedure']
    y = pd.DataFrame(df.pop('ExpiredHospital'))
    x1 = df.drop(labels=categorical_features, axis=1)
    x1.fillna(np.floor(x1.mean()), inplace=True)
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

    rf = RandomForestClassifier(
        class_weight={0: 1, 1: scale}
    )

    param_dist = {
        'n_estimators': randint(100, 200),
        'max_depth': randint(15, 25),
        'min_samples_leaf': randint(1, 5),
        'min_samples_split': randint(1, 8),
    }

    rand_params = RandomizedSearchCV(rf,
                                     scoring="f1_macro",
                                     param_distributions=param_dist,
                                     n_iter=50,
                                     cv=5,
                                     n_jobs=4,
                                     verbose=3,
                                     )

    rand_params.fit(x_train, y_train.values.ravel())
    best_params = rand_params.best_params_
    print('Best hyperparameters:', best_params)
    best_rf = RandomForestClassifier(
        **best_params,
        class_weight={0: 1, 1: scale},
    )
    best_rf.fit(x_train, y_train.values.ravel())
    y_pred = best_rf.predict(x_val)
    print(classification_report(y_val, y_pred))
    # joblib.dump(best_rf, 'rf_classifier.joblib')

def predict(df):
    x, _ = preprocessing(df)
    loaded_rf = joblib.load('rf_classifier.joblib')
    y_pred = loaded_rf.predict(x)
    pd.DataFrame(y_pred).to_csv('RF-preds.csv')
    print('Saved!')

if __name__ == '__main__':
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    # Current best: 0.9379821958456973
    train(df)
    # predict(pd.read_csv('Assignment3-Unknown-Dataset.csv'))









# Data Processing
import pandas as pd
import numpy as np
# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
scaler = StandardScaler()
device = 'cuda'


def preprocessing(df):
    # Columns with missing value: AdmitDiagnosis, religion, marital_status, LOSgroupNum
    # Dropping marital_status, LOSgroupNum because too many missing values; AdmitDiagnosis, AdmitProcedure because too many classes.
    df.drop(labels=['LOSgroupNum'], axis=1, inplace=True)
    categorical_features = ['gender', 'admit_type', 'admit_location', 'insurance', 'religion', 'ethnicity', 'AdmitDiagnosis', 'AdmitProcedure', 'marital_status']
    y = pd.DataFrame(df.pop('ExpiredHospital'))
    x1 = df.drop(labels=categorical_features, axis=1)
    # print(np.floor(x1.mean()))
    # x1.fillna(np.floor(x1.mean()), inplace=True)
    x2 = df[categorical_features]
    # x2 = x2.fillna(x2.mode().iloc[0])
    x2 = x2.apply(lambda x: pd.factorize(x)[0])
    data = pd.concat([x1, x2], axis=1)
    return data, y


def train(df):
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    x_train, y_train = preprocessing(train_df)
    x_val, y_val = preprocessing(val_df)
    scale = df['ExpiredHospital'].value_counts()[0]/df['ExpiredHospital'].value_counts()[1]

    model = DecisionTreeClassifier(
        class_weight={0: 1, 1: scale}
    )

    param_dist = {
        'max_depth': randint(10, 50),
        'min_samples_leaf': randint(2, 8),
        'min_samples_split': randint(2, 8),
    }

    rand_params = RandomizedSearchCV(
        model,
        scoring="f1_macro",
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        n_jobs=-1,
        verbose=3,
    )

    rand_params.fit(x_train, y_train.values.ravel())
    best_params = rand_params.best_params_
    print('Best hyperparameters:', best_params)
    best_model = DecisionTreeClassifier(
        **best_params,
    )

    best_model.fit(x_train, y_train.values.ravel())
    y_pred = best_model.predict(x_val)
    print(classification_report(y_val, y_pred))
    return model


def predict(df):
    x, _ = preprocessing(df)
    loaded_rf = joblib.load('lgbm_classifier.joblib')
    y_pred = loaded_rf.predict(x)
    pd.DataFrame(y_pred).to_csv('lgbm-preds.csv')
    print('Saved!')

if __name__ == '__main__':
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    model = train(df)
    # predict(pd.read_csv('Assignment3-Unknown-Dataset.csv'))









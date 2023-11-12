# Data Processing
import pandas as pd
import numpy as np
# Modelling
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


def preprocessing(df):
    # Columns with missing value: AdmitDiagnosis, religion, marital_status, LOSgroupNum
    # Dropping religion and LOSgroupNum because SVC can't handle missing values well.
    df = df.drop(labels=['LOSgroupNum', 'religion'], axis=1)
    categorical_features = ['marital_status', 'gender', 'admit_type', 'admit_location', 'insurance', 'ethnicity', 'AdmitDiagnosis', 'AdmitProcedure']
    y = pd.DataFrame(df.pop('ExpiredHospital'))
    x1 = df.drop(labels=categorical_features, axis=1)
    x1.fillna(np.floor(x1.mean()), inplace=True)
    # Normalize numerical features
    x1 = pd.DataFrame(scaler.fit_transform(x1), columns=x1.columns)
    x2 = df[categorical_features]
    x2 = x2.fillna(x2.mode().iloc[0])
    # Integer Encode categorical features.
    x2 = x2.apply(lambda x: pd.factorize(x)[0])
    data = pd.concat([x1, x2, y], axis=1)
    return data

def train(df):
    df_preprocessed = preprocessing(df)
    train_df, val_df = train_test_split(df_preprocessed, test_size=0.8, shuffle=True)
    x_train, y_train = train_df.drop(labels=['ExpiredHospital'], axis=1), train_df['ExpiredHospital']
    x_val, y_val = val_df.drop(labels=['ExpiredHospital'], axis=1), val_df['ExpiredHospital']
    model = LinearSVC(max_iter=1000, dual=False)
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_val)
    report = classification_report(y_val, y_pred)
    print(report)
    # joblib.dump(model, 'LinSVC_classifier.joblib')

def predict(df):
    x, _ = preprocessing(df)
    loaded_rf = joblib.load('rf_classifier.joblib')
    y_pred = loaded_rf.predict(x)
    pd.DataFrame(y_pred).to_csv('RF-preds.csv')
    print('Saved!')

if __name__ == '__main__':
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    train(df)
    # predict(pd.read_csv('Assignment3-Unknown-Dataset.csv'))









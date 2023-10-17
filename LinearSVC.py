# Data Processing
import pandas as pd
import numpy as np
# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

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
    train_df, val_df = train_test_split(df, test_size=0.8, shuffle=True)
    x_train, y_train = preprocessing(train_df)
    x_val, y_val = preprocessing(val_df)
    model = LinearSVC(max_iter=1000, dual=False)
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    print(f'Accuracy: {accuracy} | Recall: {recall}')
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









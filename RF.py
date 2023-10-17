# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import export_graphviz
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz


def preprocessing(df):
    # Columns with missing value: AdmitDiagnosis, religion, marital_status, LOSgroupNum
    # Dropping marital_status, LOSgroupNum because too many missing values; AdmitDiagnosis, AdmitProcedure because too many classes.
    df.drop(labels=['marital_status', 'LOSgroupNum'], axis=1, inplace=True)

    categorical_features = ['gender', 'admit_type', 'admit_location', 'insurance',
                            'religion', 'ethnicity', 'AdmitDiagnosis', 'AdmitProcedure']
    y = pd.DataFrame(df.pop('ExpiredHospital'))
    x1 = df.drop(labels=categorical_features, axis=1)
    x1.fillna(np.floor(x1.mean()), inplace=True)
    x2 = df[categorical_features]
    x2 = x2.fillna(x2.mode().iloc[0])
    x2 = x2.apply(lambda x: pd.factorize(x)[0])
    data = pd.concat([x1, x2], axis=1)
    return data, y


if __name__ == '__main__':
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)
    x_train, y_train = preprocessing(train_df)
    x_val, y_val = preprocessing(val_df)
    rf = RandomForestClassifier()

    # rf.fit(x_train, y_train.values.ravel())
    # y_pred = rf.predict(x_val)
    # accuracy = accuracy_score(y_val, y_pred)
    # print("Accuracy:", accuracy)

    # Randomizing params
    param_dist = {'n_estimators': randint(120, 280),
                  'max_depth': randint(10, 25)}
    rand_params = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=5)
    rand_params.fit(x_train, y_train.values.ravel())
    best_rf = rand_params.best_estimator_
    # Print the best hyperparameters
    print('Best hyperparameters:', rand_params.best_params_)
    best_rf.fit(x_train, y_train.values.ravel())
    y_pred = best_rf.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)









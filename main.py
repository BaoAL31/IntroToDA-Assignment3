import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

class net(nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    # Columns with missing value: AdmitDiagnosis, religion, marital_status, LOSgroupNum
    # Dropping marital_status, LOSgroupNum because too many missing values; AdmitDiagnosis, AdmitProcedure because too many classes.
    df.drop(labels=['marital_status', 'LOSgroupNum', 'AdmitDiagnosis', 'AdmitProcedure'], axis=1, inplace=True)

    categorical_features = ['gender', 'admit_type', 'admit_location', 'insurance',
                            'religion', 'ethnicity']
    y = torch.tensor(df.pop('ExpiredHospital').values)

    numeric_features = df.drop(labels=categorical_features, axis=1)
    numeric_features.fillna(numeric_features.mean())
    x1 = torch.tensor(numeric_features.values)
    df = df.fillna(df.mode().iloc[0])
    x2 = df[categorical_features]
    x2 = torch.tensor(x2.apply(lambda x: pd.factorize(x)[0]).values)




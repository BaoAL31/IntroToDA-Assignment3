import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class Net(nn.Module):
    def __init__(self, class_counts):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=class_count, embedding_dim=20) for class_count in class_counts])
        self.bn1 = nn.BatchNorm1d(16)
        self.ln1 = nn.Linear(17, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.ln2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.emb_drop = nn.Dropout(p=0.5)
        self.emb_lin = nn.Linear(160, 1)

    def forward(self, x1, x2):
        x_cat = [e(x2[:, i]) for i, e in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, 1)
        x_cat = self.emb_lin(x_cat)
        x_cat = self.emb_drop(x_cat)
        x_num = self.bn1(x1.float())
        x = torch.cat([x_num, x_cat], 1)
        x = F.relu(self.ln1(x))
        x = self.bn2(x)
        x = F.relu(self.ln2(x))
        # x = self.bn3(x)
        x = self.out(x)
        return x


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
    return x1, x2, y

loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))

def get_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = acc * 100
    return acc

def get_recall(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_pos_preds = 0

    for i in range(y_pred_tag.shape[0]):
        if y_pred_tag[i] == y_test[i] == 1:
            correct_pos_preds += 1
    positive_sum = y_test.sum().float()
    return correct_pos_preds/positive_sum


def train_model(model, optim, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        batch_size = y.shape[0]
        output = model(x1, x2)
        loss = loss_fn(output.squeeze(), y.squeeze().float())
        model.zero_grad()
        loss.backward()
        optim.step()
        total += batch_size
        sum_loss += batch_size * (loss.item())
    return sum_loss/total

def val_loss(model, valid_dl):
    model.eval()
    total = 0
    total_acc = 0
    for x1, x2, y in valid_dl:
        batch_size = y.shape[0]
        output = model(x1, x2)
        total_acc += batch_size * get_acc(output.squeeze(), y.squeeze().float())
        print(f'Recall: {get_recall(output.squeeze(), y.squeeze().float())}')
        loss = loss_fn(output.squeeze(), y.squeeze().float())
        total += batch_size
        pred = torch.max(output, 1)[1]
    print(f'Accuracy: {total_acc / total}')
    return

class HospitalDataset(Dataset):
    def __init__(self, x1, x2, y):
        self.x_num = torch.tensor(x1)  # numerical columns
        self.x_cat = torch.tensor(x2.values)  # categorical columns
        self.y = torch.tensor(y.values)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


if __name__ == '__main__':
    file = 'Assignment3-Healthcare-Dataset.csv'
    df = pd.read_csv(file)
    train_df, val_df = train_test_split(df, test_size=0.1, shuffle=False)
    x1_train, x2_train, y_train = preprocessing(train_df)
    x1_train = scaler.fit_transform(x1_train)

    x1_val, x2_val, y_val = preprocessing(val_df)
    x1_val = scaler.fit_transform(x1_val)

    train_ds = HospitalDataset(x1_train, x2_train, y_train)
    val_ds = HospitalDataset(x1_val, x2_val, y_val)

    batch_size = 15000
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    # print(val_ds[0])
    # for x in val_ds:
    #     print(x)
    class_counts = x2_train.nunique().values
    net = Net(class_counts)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=0.015, weight_decay=0.00001)
    epochs = 200

    for i in range(epochs):
        loss = train_model(net, optim, train_dl)
        print(f"Epoch: {i} | Loss: {loss}")
        val_loss(net, val_dl)





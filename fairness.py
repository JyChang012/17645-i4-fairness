import pandas as pd
import numpy as np
from torch import nn
import torch
from torch.nn import Embedding, Module, ModuleDict, Sequential, Linear, BatchNorm1d, ReLU
from typing import Dict, List, Set
import torch.optim as optim
from sklearn.metrics import roc_curve
import seaborn as sas
from sklearn.metrics import auc, confusion_matrix, accuracy_score

sas.set_theme()
from matplotlib import pyplot as plt

torch.manual_seed(15213)
np.random.seed(15213)


@torch.no_grad()
def evaluate(model: Module, val_df: pd.DataFrame, threshold=.5):
    model.eval()
    y_ = model(val_df.iloc[:, :-1])
    y_ = torch.sigmoid(y_)
    prob = y_.numpy().copy()
    y_[y_ < threshold] = 0
    y_[y_ >= threshold] = 1
    y_ = y_.numpy().astype(int).flatten()
    gt = val_df.iloc[:, -1].cat.codes.to_numpy(dtype=int)
    acc = np.sum(y_ == gt) / len(y_)
    return prob, gt, y_, acc


def train(model, train_df: pd.DataFrame, epoch=200, lr=1e-2):
    print('training start')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epoch):
        optimizer.zero_grad(set_to_none=True)
        out = model(train_df.iloc[:, :-1])
        loss = criterion(out, torch.tensor(train_df.iloc[:, -1].cat.codes.to_numpy(), dtype=torch.float))
        loss.backward()
        print(loss.item())
        optimizer.step()


class LinearBNReLU(Sequential):

    def __init__(self, in_n, out_n):
        super().__init__(
            Linear(in_n, out_n),
            BatchNorm1d(out_n),
            ReLU()
        )


class MLP(Module):

    def __init__(
        self, unq_n: Dict[str, int], cont_col: List[str]
        , emb_dim=2, layers_n=(5, 5, 5, 5), exclude: Set[str] = set()
    ):
        super(MLP, self).__init__()
        unq_n = {name: val for name, val in unq_n.items() if name not in exclude}
        cont_col = list(filter(lambda x: x not in exclude, cont_col))

        self.embs = ModuleDict({name: Embedding(n, emb_dim) for name, n in unq_n.items()})
        inp_size = len(unq_n) * emb_dim + len(cont_col)
        layers_n = [inp_size] + list(layers_n)
        linears = [LinearBNReLU(i, j) for i, j in zip(layers_n, layers_n[1:])]

        self.mlp = Sequential(*linears)
        self.out = Linear(layers_n[-1], 1)

        self.cont_col = cont_col
        self.dis_col = list(unq_n.keys())

    def forward(self, x: pd.DataFrame):
        dis = []
        for dcol in self.dis_col:
            y = torch.tensor(x[dcol].cat.codes.to_numpy(), dtype=torch.int)
            emb = self.embs[dcol]
            y[y == -1] = emb.weight.shape[0] - 1
            dis.append(emb(y))
        dis = torch.hstack(dis)
        cont = torch.tensor(x.loc[:, self.cont_col].to_numpy(), dtype=torch.float)
        x = torch.hstack([dis, cont])
        x = self.mlp(x)
        return self.out(x).flatten()


def get_col_id(cols):
    return [i for i, v in enumerate(cols) if v]


def main():
    df = pd.read_csv("./input/german-credit-data-with-risk/german_credit_data.csv", index_col=0)
    discrete_type = np.dtype('O')
    cont_type = np.dtype('int64')
    dfx = df.iloc[:, :-1]
    discrete_cols = list(map(lambda x: x == discrete_type, dfx.dtypes))
    discrete_cols_names = dfx.columns[discrete_cols]
    cont_cols = list(map(lambda x: x == cont_type, dfx.dtypes))
    cont_cols_names = dfx.columns[cont_cols]
    for col in discrete_cols_names:
        df[col] = pd.Categorical(dfx[col])
    df.iloc[:, -1] = pd.Categorical(df.iloc[:, -1])

    N = df.shape[0]
    train_df = df.iloc[:int(N * .8)]
    val_df = df.iloc[int(N * .8):]

    model = MLP({k: len(df[k].unique()) for k in discrete_cols_names}, cont_cols_names, emb_dim=2, layers_n=(7,))

    train(model, train_df)

    prob, gt, y_, acc = evaluate(model, val_df)

    fpr, tpr, thresholds = roc_curve(gt, prob)
    auc_score = auc(fpr, tpr)
    print('baseline model')
    print(f'auc is {auc_score}')

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('roc.jpg')
    plt.show()

    print(f'acc is {acc}')

    # torch.save(model.state_dict(), 'model.pkl')

    # anti class for gender
    val_df_male = val_df.copy()
    val_df_male['Sex'] = pd.Categorical(['male'] * 200, categories=val_df_male['Sex'].cat.categories)
    val_df_female = val_df.copy()
    val_df_female['Sex'] = pd.Categorical(['female'] * 200, categories=val_df_male['Sex'].cat.categories)

    _, _, y_male, _ = evaluate(model, val_df_male)
    _, _, y_female, _ = evaluate(model, val_df_female)
    inconsist = np.sum(y_male != y_female) / y_male.shape[0]
    print(f'ANTI-CLASSIFICATION for gender: Percentage of inconsistencies: {inconsist * 100:2f}%')

    # anti class for age
    quarter_1 = df['Age'].quantile(1 / 4)
    quarter_3 = df['Age'].quantile(3 / 4)

    print(f'young {quarter_1} vs old {quarter_3}')

    val_df_old = val_df.copy()
    val_df_old['Age'] = quarter_3
    val_df_young = val_df.copy()
    val_df_young['Age'] = quarter_1

    _, _, y_old, _ = evaluate(model, val_df_old)
    _, _, y_young, _ = evaluate(model, val_df_young)
    inconsist = np.sum(y_old != y_young) / y_young.shape[0]
    print(f'ANTI-CLASSIFICATION for age: Percentage of inconsistencies: {inconsist * 100:2f}%')

    # group fair
    # gender
    _, gt_male, y_male, _ = evaluate(model, val_df.loc[val_df['Sex'] == 'male'])
    p_male = y_male.sum() / y_male.shape[0]
    _, gt_female, y_female, _ = evaluate(model, val_df.loc[val_df['Sex'] == 'female'])
    p_female = y_female.sum() / y_female.shape[0]

    print('group fairness')
    print(f'for male: {p_male}, for female {p_female}, diff = {p_male - p_female}')

    # age
    mean = df['Age'].mean()
    _, gt_old, y_old, _ = evaluate(model, val_df.loc[val_df['Age'] >= mean])
    p_old = y_old.sum() / y_old.shape[0]
    _, gt_young, y_young, _ = evaluate(model, val_df.loc[val_df['Age'] < mean])
    p_young = y_young.sum() / y_young.shape[0]

    print('group fairness')
    print(f'for old: {p_old}, for young {p_young}, diff = {p_old - p_young}')

    # separation
    # gender
    conf_male = confusion_matrix(gt_male, y_male).astype(float)
    conf_male /= np.sum(conf_male, axis=1, keepdims=True)
    fpr_male, fnr_male = conf_male[0, 1], conf_male[1, 0]

    conf_female = confusion_matrix(gt_female, y_female).astype(float)
    conf_female /= np.sum(conf_female, axis=1, keepdims=True)
    fpr_female, fnr_female = conf_female[0, 1], conf_female[1, 0]

    print('Separation')
    print(f'fpr: male {fpr_male}, female {fpr_female}, diff = {fpr_male - fpr_female}')
    print(f'fnr: male {fnr_male}, female {fnr_female}, diff = {fnr_male - fnr_female}')

    conf_old = confusion_matrix(gt_old, y_old).astype(float)
    conf_old /= np.sum(conf_old, axis=1, keepdims=True)
    fpr_old, fnr_old = conf_old[0, 1], conf_old[1, 0]

    conf_young = confusion_matrix(gt_young, y_young).astype(float)
    conf_young /= np.sum(conf_young, axis=1, keepdims=True)
    fpr_young, fnr_young = conf_young[0, 1], conf_young[1, 0]

    print(f'fpr: old {fpr_old}, young {fpr_young}, diff = {fpr_old - fpr_young}')
    print(f'fnr: old {fnr_old}, young {fnr_young}, diff = {fnr_old - fnr_young}')

    # improve on gender
    # anti class
    fair_model = MLP({k: len(df[k].unique()) for k in discrete_cols_names}, cont_cols_names, emb_dim=2, layers_n=(7,),
                     exclude={'Sex'})

    train(fair_model, train_df)
    _, gt, y_, _ = evaluate(fair_model, val_df_male)
    acc_fair = accuracy_score(gt, y_)
    print(f'fair model acc: {acc_fair}')
    _, _, y_male, _ = evaluate(fair_model, val_df_male)
    _, _, y_female, _ = evaluate(fair_model, val_df_female)
    inconsist = np.sum(y_male != y_female) / y_male.shape[0]
    print(f'Fair model: ANTI-CLASSIFICATION for gender: Percentage of inconsistencies: {inconsist * 100:2f}%')

    # group fair
    _, gt_male, y_male, _ = evaluate(model, val_df.loc[val_df['Sex'] == 'male'], threshold=.53)
    p_male = y_male.sum() / y_male.shape[0]
    _, gt_female, y_female, _ = evaluate(model, val_df.loc[val_df['Sex'] == 'female'], threshold=.47)
    p_female = y_female.sum() / y_female.shape[0]

    print('Improve group fairness')
    fair_acc = accuracy_score(np.concatenate([gt_male, gt_female]), np.concatenate([y_male, y_female]))
    print(f'acc after changing threshold for group: {fair_acc}')
    print(f'for male: {p_male}, for female {p_female}, diff = {p_male - p_female}')

    # separation
    _, gt_male, y_male, _ = evaluate(model, val_df.loc[val_df['Sex'] == 'male'], threshold=.52)
    _, gt_female, y_female, _ = evaluate(model, val_df.loc[val_df['Sex'] == 'female'], threshold=.50)


    conf_male = confusion_matrix(gt_male, y_male).astype(float)
    conf_male /= np.sum(conf_male, axis=1, keepdims=True)
    fpr_male, fnr_male = conf_male[0, 1], conf_male[1, 0]

    conf_female = confusion_matrix(gt_female, y_female).astype(float)
    conf_female /= np.sum(conf_female, axis=1, keepdims=True)
    fpr_female, fnr_female = conf_female[0, 1], conf_female[1, 0]

    print('Improve separation')
    fair_acc = accuracy_score(np.concatenate([gt_male, gt_female]), np.concatenate([y_male, y_female]))
    print(f'acc after changing threshold for separation: {fair_acc}')
    print(f'fpr: male {fpr_male}, female {fpr_female}, diff = {fpr_male - fpr_female}')
    print(f'fnr: male {fnr_male}, female {fnr_female}, diff = {fnr_male - fnr_female}')



if __name__ == '__main__':
    main()

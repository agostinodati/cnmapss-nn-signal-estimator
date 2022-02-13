import pandas as pd
from pandas import DataFrame

X = pd.read_pickle('dataset/train_X.pkl')
y = pd.read_pickle('dataset/train_y.pkl')
X1 = pd.read_pickle('dataset/train_X_1.pkl')
y1 = pd.read_pickle('dataset/train_Y_1.pkl')

X = X.append(X1)
y = y.append(y1)

X.to_pickle("dataset/train_X_merged.pkl")
y.to_pickle("dataset/train_Y_merged.pkl")
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import load


def windowgenerator(dataframe, target, shift):
    Y = np.array(df[target])
    # print("Y", Y)

    for i in reversed(range(1, shift+1)):
        if i == (shift):
            X = Y[(shift-i):(len(Y)-i)]
        else:
            X2 = Y[(shift-i):(len(Y)-i)]
            X = np.concatenate([X, X2])
    Y = Y[shift:len(dataframe)]
    return X, Y


def preprocessing(df, n=5):
    # loc_all = df[["lat", "lng"]]
    # stand = StandardScaler().fit(loc_all)
    # df[["lat", "lng"]] = stand.transform(df[["lat", "lng"]])

    for i in range(1, n+1):
        df[['lat_lag{}'.format(i), 'lng_lag{}'.format(i)]
           ] = df[["lat", "lng"]].shift(i)

    df = df.dropna()
    Y = df[["lat", "lng"]].to_numpy()
    X = df.drop(["datetime", "lat", "lng"], axis=1).to_numpy()
    print(X.shape)

    _l = len(X)
    _X = X.reshape(_l, n, 2)
    # print(_X, Y)

    return _X, Y

if __name__ == "__main__":
    df = pd.read_csv('DUJUAN_trace.csv')
    preprocessing(df)

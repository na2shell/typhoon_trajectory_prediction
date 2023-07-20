# from tensorflow.keras.layers import Dropout, Conv1D, MaxPool1D, Flatten

# from keras.layers.convolutional import Conv1D, UpSampling1D
# from keras.layers.pooling import MaxPooling1D
from keras.layers import Dense, Dropout, Conv1D, Flatten
from keras import Model
from keras.layers import Input, LSTM, Dense, Concatenate, MaxPooling1D
import LSTM_preprocessing as pp
from keras.optimizers import Adam
import glob as glob

import pandas as pd
import numpy as np
from pickle import load

window = 6

X_train = np.empty([0, window, 2])
X_test = np.empty([0, window, 2])
Y_train = np.empty([0, 2])
Y_test = np.empty([0, 2])

for name in glob.glob('trace_data/*.csv'):
    df = pd.read_csv(name)
    X, Y = pp.preprocessing(df,window)
    # print(X.shape)
    # print(Y.shape)
    i = int(len(X)*0.8)
    _X_train = X[:i]
    _Y_train = Y[:i]

    X_train = np.concatenate([X_train,_X_train])
    Y_train = np.concatenate([Y_train,_Y_train])

    if "DUJUAN" in name:
        X_test = X[i:i+1]
        Y_test = Y[i:i+1]
        # stand = _stand
        print("X_test",X_test)

# print("hoge")
# print(X_train)

# ################ CNN part ####################
# inputs = Input(shape=(X.shape[1], X.shape[2]))
# conv1th = Conv1D(filters=5, kernel_size=3)(inputs)
# drop1 = Dropout(0.2)(conv1th)
# maxpool = MaxPool1D(pool_size=2)(drop1)
# ################ CNN part ####################

################ LSTM part ####################
inputs2 = Input(shape=(X.shape[1], X.shape[2]))
lstm = LSTM(window, return_sequences=True)(inputs2)
drop2 = Dropout(0.2)(lstm)
maxpool2 = MaxPooling1D(pool_size=3)(drop2)
################ LSTM part ####################

# ################ fusion part ####################
# conc = Concatenate(axis=1)([maxpool, maxpool2])
# flat = Flatten()(conc)
# ################ fusion part ####################
flat = Flatten()(maxpool2)

################ last part ####################
dense1 = Dense(64, activation="relu")(flat)
outputs = Dense(2)(dense1)
################ last part ####################

################ model compile part ####################
# model = keras.Model(inputs=[inputs, inputs2], outputs=outputs)
model = Model(inputs=inputs2, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.01),
              loss="mse",
              metrics="mae")



# print(X_train,Y_train)
model.fit(X_train,
          Y_train,
          validation_split=0.1,
          epochs=200,
          verbose=2)



stand = load(open("sample.pkl", "rb"))

pred = model.predict(X_test)
pred_inv = stand.inverse_transform(pred)
Y_test_inv = stand.inverse_transform(Y_test)
X_test_inv = stand.inverse_transform(X_test[0])
# print(pred_inv, Y_test_inv)

Y_train_inv = stand.inverse_transform(Y_train)

print(pred_inv.shape)
print(X_test.shape)
predict = np.concatenate([X_test_inv[::-1], pred_inv])
actual = np.concatenate([X_test_inv[::-1], Y_test_inv])
print("hoge",actual)
print("hoge",predict)

ddf = pd.DataFrame(predict, columns=["lat","lng"])
ddf.to_csv("sample_result_predict.csv",index=False)

ddf = pd.DataFrame(actual, columns=["lat","lng"])
ddf.to_csv("sample_result_actual.csv",index=False)
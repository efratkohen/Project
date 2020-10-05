import clean_data_svi as cds
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

def series_to_supervised(sequences, n_steps_in=1, n_steps_out=1, jump=1, binary=False):
    X, Y = list(), list()

    for i in range(0, len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[out_end_ix - 1:out_end_ix][:, 0]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

def normalize(X, Y):
    scalers = list()
    for i in tqdm(range(X.shape[1])):
        scalers.append(MinMaxScaler())
        X[:, i, :] = scalers[i].fit_transform(X[:, i, :])

    scalers.append(MinMaxScaler())
    Y = scalers[i + 1].fit_transform(Y)
    return X, Y, scalers

# if __name__ == "__main__":
#     svi_df = pd.read_csv("clean_tables/svi_1_tem.csv", index_col="date")
#     micro_df = pd.read_csv("clean_tables/micro_1.csv", index_col="date")
#     svi_df.index = pd.to_datetime(svi_df.index, dayfirst=True)
#     micro_df.index = pd.to_datetime(micro_df.index, dayfirst=True)
#     join = pd.concat([svi_df, micro_df], axis=1)
#     join = join.fillna(method='ffill').dropna(axis=0, how='any')
#     join_arr = join.values
#     X, Y = series_to_supervised(join_arr)
#     X_normalize, Y_normalize, scalers = normalize(X, Y)
#     X_train, X_test, y_train, y_test = train_test_split(X_normalize, Y_normalize, test_size=0.25, random_state=42)
#     model = Sequential()
    # model.fit(Xtrain, Ytrain, epochs=30, batch_size=batch_32, shuffle=True)
    # model.add(LSTM(units=100, activation='relu', name='first_lstm', recurrent_dropout=0.1, input_shape=(Xtrain.shape[1], 1)))
    # model.add(Dense(1, activation="linear"))cd
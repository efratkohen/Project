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
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


def series_to_supervised(sequences, n_steps_in=1, n_steps_out=1, jump=1, binary=False):
    """create array for x and y 
    n_steps_in - how many rows to take before this row for evaluation- window size
    n_steps_out - prediction ahead 
    """
    X, Y = list(), list()
    svi_limit = 150

    for i in range(0, len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = (
            sequences[i:end_ix, :],
            sequences[out_end_ix - 1 : out_end_ix][:, 0],
        )

        if binary:
            seq_y = [float(seq_y < svi_limit)]

        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)


def normalize(X, Y, feature_range=(0, 1)):
    scalers = list()
    for i in tqdm(range(X.shape[1])):
        scalers.append(MinMaxScaler(feature_range))
        X[:, i, :] = scalers[i].fit_transform(X[:, i, :])

    scalers.append(MinMaxScaler())
    Y = scalers[i + 1].fit_transform(Y)
    return X, Y, scalers


def create_join_x_y_arr(
    reactor_list: list, n_steps_in=1, n_steps_out=1, jump=1, binary=False
):
    for i in range(len(reactor_list)):
        if i == 0:
            X, Y = series_to_supervised(
                reactor_list[i].values, n_steps_in, n_steps_out, jump, binary
            )
        else:
            X1, Y1 = series_to_supervised(
                reactor_list[i].values, n_steps_in, n_steps_out, jump, binary
            )
            X = np.concatenate((X, X1), axis=0)
            Y = np.concatenate((Y, Y1), axis=0)
    return X, Y


def evaluate(model, Xtest, Ytest, scalers, binary=False):
    Yhat = model.predict(Xtest)
    if not binary:
        Yhat = scalers[-1].inverse_transform(Yhat)
        Ytest = scalers[-1].inverse_transform(Ytest)
    return Yhat, Ytest


def results(y_real, y_predict, binary=True):
    if binary:
        cm = confusion_matrix(y_real, y_predict)
        tn, fp, fn, tp = confusion_matrix(y_real, y_predict).ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        TNR = (tn) / (tn + fp)
        NPV = (tn) / (tn + fn)
        f1 = 2 * (TNR * NPV) / (TNR + NPV)
        return accuracy, TNR, NPV, f1
    else:
        rmse = sqrt(mean_squared_error(y_real, y_predict))
        return rmse


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


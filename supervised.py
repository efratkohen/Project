from typing import Union, Tuple
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def drop_nan_rows(microscopic_df_list: list):
    """Drop nan rows from microscopic dataframe"""

    for i in range(4):
        microscopic_df_list[i] = microscopic_df_list[i].dropna(thresh=10)
    return microscopic_df_list


def merge_data(first_col: int, last_col: int, data_micro: pd.DataFrame, data_svi: pd.DataFrame, hyper_param_day: int) -> pd.DataFrame:
    """join data from microscopic with labeled svi and sv.
    first_col is the first column from microscopic df that going to enter the ml model
    last_col is the last column from microscopic df that going to enter the ml model
    hyper_param_day is number of days in delay to take the labeled data
    
    Returns
    -------
    join_df: data frame of join data
    """

    join_df = data_micro.iloc[:, np.r_[0, first_col:last_col]]
    join_df.loc[:, 'Settling_velocity'], join_df.loc[:, 'SVI'], join_df.loc[:, 'SV_label'], join_df.loc[:, 'SVI_label']   = "init" , "init", "init", "init"
    dates = join_df["date"].to_list()
    l = join_df.shape[0]
    for row in range(l):
        for col in range (1,5): 
            if(data_svi.loc[data_svi['date']==dates[row]].index[0] + hyper_param_day) < data_svi.shape[0]:
                join_df.iloc[row, col + last_col - first_col] = data_svi.iloc[data_svi.loc[data_svi['date']==dates[row]].index[0] + hyper_param_day,col]
    join_df = join_df[join_df.SV_label != 'init']
    return join_df


def merge_data_df_list(first_col: int, last_col: int, micro_df_list: list, svi_df_list: list, hyper_param_day: int) -> list:
    join_df_list = []
    for i in range(4):
        reactor_df = merge_data(first_col, last_col, micro_df_list[i], svi_df_list[i], hyper_param_day)
        join_df_list.append(reactor_df)
    return join_df_list


def check_K_values(k_max: int, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    k_range = range(1,k_max)
    scores_k = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores_k.append(knn.score(X_test, y_test))
    scores_data = pd.DataFrame({"k": k_range, "score": scores_k})
    return scores_data
    
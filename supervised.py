from typing import Union, Tuple
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

def drop_nan_rows(microscopic_df_list: list):
    """Drop nan rows from microscopic dataframe"""

    for i in range(0,4):
        microscopic_df_list[i] = microscopic_df_list[i].dropna(thresh=10)
    return microscopic_df_list


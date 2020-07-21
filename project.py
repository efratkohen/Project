from typing import Union, Tuple
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_file(data_fname: Union[pathlib.Path, str]):
    """Check for valid file
    accept strings and pathlib.Path objects"""

    try:
        fname = pathlib.Path(data_fname)
    except TypeError:
        print("ERROR: Please supply a string or a pathlib.Path instance.")
        raise
    if not fname.exists():
        raise ValueError(f"File {str(fname)} doesn't exist.")
    return fname


def read_data(data_fname: Union[pathlib.Path, str]) -> pd.DataFrame:
    """Reads data into DF"""

    data = pd.read_csv(data_fname)
    return data


def check_SVI_values(data: pd.DataFrame) -> pd.DataFrame:
    """check and replace incorrect values with nan"""

    data = data.replace(0, np.nan)
    data.loc[:, data.columns.str.contains("SV")] = data.loc[
        :, data.columns.str.contains("SV")
    ].apply(lambda x: [y if 0 < y < 6 else np.nan for y in x])
    data.loc[:, data.columns.str.contains("volume")] = data.loc[
        :, data.columns.str.contains("volume")
    ].apply(lambda x: [y if 0 < y < 1000 else np.nan for y in x])
    data.loc[:, data.columns.str.contains("mlss")] = data.loc[
        :, data.columns.str.contains("mlss")
    ].apply(lambda x: [y if 500 < y < 4000 else np.nan for y in x])
    return data


def SVI_calculate(data: pd.DataFrame) -> pd.DataFrame:
    """Add column of SVI caculation fot each reactor"""

    data["SVI1"] = data["volume reactor 1"] * 1000 / data["mlss reactor 1"]
    data["SVI2"] = data["volume reactor 2"] * 1000 / data["mlss reactor 2"]
    data["SVI3"] = data["volume reactor 3"] * 1000 / data["mlss reactor 3"]
    data["SVI4"] = data["volume reactor 4"] * 1000 / data["mlss reactor 4"]
    return data


def split_to_reactor(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the SVI data frame to 4 dataframes for each reactor.
    Change the columns names to be identical
    
    Returns
    -------
    data frame
        data_reactor1, data_reactor2, data_reactor3, data_reactor4
    """

    data_reactor1 = data[["Date", "SV reactor 1", "SVI1"]]
    data_reactor2 = data[["Date", "SV reactor 2", "SVI2"]]
    data_reactor3 = data[["Date", "SV reactor 3", "SVI3"]]
    data_reactor4 = data[["Date", "SV reactor 4", "SVI4"]]
    df_list=[data_reactor1, data_reactor2, data_reactor3, data_reactor4]
    for df in df_list:
        df.columns= ['date', 'Settling_velocity', 'SVI']
    return data_reactor1, data_reactor2, data_reactor3, data_reactor4


def label_data(data: pd.DataFrame, label_SVI: list, label_SV: list) -> pd.DataFrame:
    """add labels column for SVI and SV results as bad, reasonable or good"""

    data['SV_label'] = np.where(data.loc[:, 'Settling_velocity']<=label_SV[0], 'bad',
         np.where(data.loc[:, 'Settling_velocity']<=label_SV[1], 'reasonable', 'good'))
    data['SVI_label'] = np.where(data.loc[:, 'SVI']>=label_SVI[0], 'bad',
         np.where(data.loc[:, 'SVI']>=label_SVI[1], 'reasonable', 'good'))
    return data


# def add_micro_to_reactor_df(data: pd.DataFrame, data_reactor1: pd.DataFrame, data_reactor2: pd.DataFrame, data_reactor3: pd.DataFrame, data_reactor4: pd.DataFrame,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """Split the microscopic data frame and add the data to 4 dataframes for each reactor.
#     Change the columns names to be identical
    
#     Returns
#     -------
#     data frame
#         data_reactor1, data_reactor2, data_reactor3, data_reactor4
#     """

#     reactor1_col = [col for col in data.columns if '1' in col]

#     data_reactor1 = data[["Date", "SV reactor 1", "SVI1"]]
#     data_reactor2 = data[["Date", "SV reactor 2", "SVI2"]]
#     data_reactor3 = data[["Date", "SV reactor 3", "SVI3"]]
#     data_reactor4 = data[["Date", "SV reactor 4", "SVI4"]]
#     df_list=[data_reactor1, data_reactor2, data_reactor3, data_reactor4]
#     for df in df_list:
#         df.columns= ['date', 'Settling_velocity', 'SVI']
#     return data_reactor1, data_reactor2, data_reactor3, data_reactor4
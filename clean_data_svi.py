import pandas as pd 
import numpy as np

def check_svi_values_range(svi_data: pd.DataFrame) -> pd.DataFrame:
    '''
    check and replace incorrect values with nan
    '''

    svi_data.replace(0, np.nan, inplace=True)
    ranges_lst = [(0, 6), (0, 1000), (500, 4000)]
    col_names_lst = ["SV", "volume", "mlss"]

    for i in range(3):
        name = col_names_lst[i]
        low = ranges_lst[i][0]
        high = ranges_lst[i][1]
        svi_data.loc[:, svi_data.columns.str.contains(name)] = svi_data.loc[
            :, svi_data.columns.str.contains(name)
        ].apply(lambda x: [value if low < value < high else np.nan for value in x])

    return svi_data


def svi_calculate(data_svi: pd.DataFrame) -> pd.DataFrame:
    """Add column of SVI caculation fot each reactor"""
    for i in range(1, 5):
        data_svi.loc[:,f"SVI{i}"] = data_svi[f"volume reactor {i}"] * 1000 / data_svi[f"mlss reactor {i}"]

    return data_svi


def set_datetime_index(df_list: list):
    for i in range(4):
        df_list[i].set_index('date', inplace=True)
        df_list[i].index = pd.to_datetime(df_list[i].index)


def interpolate_svi_dfs(svi_df_list: list):
    for i in range(4):
        svi_df_list[i].interpolate(inplace=True, method='time')


def svi_label(svi_df_list: list):
    '''
    '''
    # define borders between bad / reasonable / good results
    SVI_label=[160.0, 120.0]
    SV_label=[3.0, 3.5] 
    for svi_df in svi_df_list:
        svi_df = label_data(svi_df, SVI_label, SV_label)


def label_data(data_svi: pd.DataFrame, label_SVI: list, label_SV: list) -> pd.DataFrame:
    '''
    add labels column for SVI and SV results as bad, reasonable or good
    '''

    data_svi.loc[:,"SV_label"] = np.where(
        data_svi.loc[:, "Settling_velocity"] <= label_SV[0],
        "bad",
        np.where(data_svi.loc[:, "Settling_velocity"] <= label_SV[1], "reasonable", "good"),
    )
    data_svi.loc[:,"SVI_label"] = np.where(
        data_svi.loc[:, "SVI"] >= label_SVI[0],
        "bad",
        np.where(data_svi.loc[:, "SVI"] >= label_SVI[1], "reasonable", "good"),
    )
    return data_svi
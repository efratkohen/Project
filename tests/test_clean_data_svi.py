from clean_data_svi import *
from files_process_save import *

import pytest
import math
# import datetime


def test_check_svi_values_range_zeros():
    # prepare 0 value
    data_svi = read_data("SVI.csv")
    data_svi.loc[1, "SV reactor 1"] = 0

    # check
    res = check_svi_values_range(data_svi)
    assert pd.isnull(res.loc[1, "SV reactor 1"])


def test_check_svi_values_range_ranges():
    # prepare out of range value
    data_svi = read_data("SVI.csv")
    # ranges : 'SV':(0, 6), 'volume':(0, 1000), 'mlss':(500, 4000)
    data_svi.loc[1, "SV reactor 1"] = 8
    data_svi.loc[1, "volume reactor 3"] = -5
    data_svi.loc[1, "mlss reactor 2"] = 499

    # check
    res = check_svi_values_range(data_svi)
    assert all(
        pd.isnull(
            [
                res.loc[1, "SV reactor 1"],
                res.loc[1, "volume reactor 3"],
                res.loc[1, "mlss reactor 2"],
            ]
        )
    )


def test_svi_calculate_results():
    data_svi = read_data("SVI.csv")
    res = svi_calculate(data_svi)
    assert math.isclose(res.loc[0,'SVI1'], 300 * 1000 / 1348)


def test_set_datetime_index():
    svi_df_list = svi_data_read_calculate_and_split()
    set_datetime_index(svi_df_list)
    # check all 4 dfs are DatetimeIndex
    for i in range(4):
        df = svi_df_list[i]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert 'date' not in df.columns


def test_interpolate_svi_dfs_changed_list():
    svi_df_list = svi_data_read_calculate_and_split()
    set_datetime_index(svi_df_list)

    interpolate_svi_dfs(svi_df_list)
    # check no nans left in all 4 dfs:
    for i in range(4):
        df = svi_df_list[i]
        assert ~df.isnull().values.all()


def test_label_data():
    svi_df_list = svi_data_read_calculate_and_split()
    set_datetime_index(svi_df_list)
    interpolate_svi_dfs(svi_df_list)
    df0 = svi_df_list[0]

    # prepare
    SV_label=[3.0, 3.5] 
    SVI_label=[160.0, 120.0]
    
    df0.iloc[0:3, 0] = [2, 3.2, 6]
    df0.iloc[0:3, 1] = [200, 150, 10]

    labeled_df0 = label_data(df0, SVI_label, SV_label)
    all(labeled_df0.iloc[0:3, 2] == ['bad','reasonable', 'good'])
    all(labeled_df0.iloc[0:3, 3] == ['bad','reasonable', 'good'])


def test_svi_label_changed_list():
    svi_df_list = svi_data_read_calculate_and_split()
    set_datetime_index(svi_df_list)
    interpolate_svi_dfs(svi_df_list)

    svi_label(svi_df_list)
    for i in range(4):
        assert 'SV_label' in svi_df_list[i].columns
        assert 'SVI_label' in svi_df_list[i].columns







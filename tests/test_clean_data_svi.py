from clean_data_svi import *
from files_process_save import *

import pytest


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
    
    pass


def test_set_datetime_index():
    pass


def test_interpolate_svi_dfs_changed_list():
    pass


def test_svi_label_changed_list():
    pass


def test_label_data():
    pass


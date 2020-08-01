from ml_prepare import ML_prepare
from datetime import timedelta

import pytest
import pandas as pd


def test_init_types():
    delay = 3
    data = ML_prepare(delay)

    # check types and lengths
    assert isinstance(data, ML_prepare)
    assert data._delay == delay
    assert isinstance(data._svi_lst, list)
    assert isinstance(data._micro_lst, list)


def test_init_df_lists():
    delay = 8
    data = ML_prepare(delay)

    assert len(data._svi_lst) == 4
    assert len(data._micro_lst) == 4
    for i in range(4):
        assert isinstance(data._svi_lst[i], pd.DataFrame)
        assert isinstance(data._micro_lst[i], pd.DataFrame)


def test_init_df_xy():
    delay = 5
    data = ML_prepare(delay)
    assert isinstance(data._x, pd.DataFrame)
    assert isinstance(data._y, pd.DataFrame)
    assert isinstance(data.delay_table, pd.DataFrame)


def test_indirect_read_and_index_svi_tables():
    delay = 2
    data = ML_prepare(delay)
    for i in range(4):
        assert isinstance(data._svi_lst[i].index, pd.DatetimeIndex)
        assert isinstance(data._svi_lst[i].index, pd.DatetimeIndex)


def test_get_columns_of_sections():
    delay = 2
    data = ML_prepare(delay)
    res = data.get_columns_of_sections()
    assert isinstance(res, tuple)
    for i in range(3):
        assert isinstance(res[i], list)


def test_get_partial_table_wrong_input():
    delay = 10
    data = ML_prepare(delay)

    section = "blah"
    with pytest.raises(AssertionError):
        t = data.get_partial_table(x_section=section, y_labels=True)


def test_get_partial_table_no_nans():
    delay = 10
    data = ML_prepare(delay)

    sections = ["all", "total_counts", "filaments", "various"]
    for section in sections:
        t = data.get_partial_table(x_section=section)
        assert not t.isnull().values.any()


def test_get_partial_table_x_sections_lengths():
    delay = 6
    data = ML_prepare(delay)

    sections = {"all": 27, "total_counts": 9, "filaments": 9, "various": 18}
    for section in sections:
        t = data.get_partial_table(x_section=section)
        assert t["x"].shape[1] == sections[section]


def test_get_partial_table_y_labels():
    delay = 0
    data = ML_prepare(delay)

    t1 = data.get_partial_table(x_section="all", y_labels=True)
    assert t1["y"].columns.to_list() == ["SV_label", "SVI_label"]

    t2 = data.get_partial_table(x_section="all", y_labels=False)
    assert t2["y"].columns.to_list() == ["Settling_velocity", "SVI"]


def test_indirect_create_x_y_delayed_length():
    delay = 8
    data = ML_prepare(delay)

    assert data._x.shape[0] == data._y.shape[0]


def test_find_closest_date_out_of_range():
    delay = 8
    data = ML_prepare(delay)
    date0 = data._svi_lst[0].index[-1]

    # this is out of range:
    res = data._find_closest_date(bio_reactor_i=0, date0=date0)

    assert not res  # assert res is False


def test_find_closest_date_missing_date():
    # find missing date in svi
    delay = 8
    data = ML_prepare(delay)
    date0 = data._micro_lst[0].index[0]
    missing_date = date0 + timedelta(days=data.delay)
    # remove the date
    data._svi_lst[0].drop(missing_date, inplace=True)

    # assert that this date was really changed 1 back
    found_date = data._find_closest_date(bio_reactor_i=0, date0=date0)
    assert found_date + timedelta(days=1) == missing_date


def test_indirect_create_x_y_bioreactor():
    delay = 10
    data = ML_prepare(delay)

    delay_time = timedelta(days=data.delay)

    # assert that date difference is correct
    data._y.loc["3", :].index[0] - data._x.loc["3", :].index[0] == delay_time

    # assert that out_of_range dates were removed from x as well
    assert data._micro_lst[3].shape[0] == data._x.loc["4", :].shape[0] + 1


def test_indirect_join_x_y():
    delay = 3
    data = ML_prepare(delay)
    x1 = data.delay_table.loc[:, "micro"].reset_index(drop=True)
    x2 = data.x.reset_index(drop=True)

    y1 = data.delay_table.loc[:, "svi"].reset_index(drop=True)
    y2 = data.y.reset_index(drop=True)

    assert x1.equals(x2)
    assert y1.equals(y2)

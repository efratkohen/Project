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
    
    assert len(data._svi_lst)==4
    assert len(data._micro_lst)==4
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
    for i in rnage(3):
        assert isinstance(res[i], list)


def test_get_partial_table_wrong_input():
    delay = 10
    data = ML_prepare(delay)

    section = 'blah'
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
    
    sections = {"all":27, "total_counts":9, "filaments":9, "various":18}
    for section in sections:
        t = data.get_partial_table(x_section=section)
        assert t['x'].shape[1]== sections[section]


def test_get_partial_table_y_labels():
    delay = 0
    data = ML_prepare(delay)

    t1 = data.get_partial_table(x_section='all', y_labels=True)
    assert t1['y'].columns.to_list() == ['SV_label','SVI_label']

    t2 = data.get_partial_table(x_section='all', y_labels=False)
    assert t2['y'].columns.to_list() == ['Settling_velocity', 'SVI']


def test_indirect_create_x_y_delayed_length():
    delay = 8
    data = ML_prepare(delay)

    assert data._x.shape[0] == data._y.shape[0]


def test_indirect_create_x_y_bioreactor():
    delay = 10
    data = ML_prepare(delay)

    delay_time = timedelta(days=data.delay)

    # assert that date difference is correct
    data._y.loc['3',:].index[0] - data._x.loc['3',:].index[0] == delay_time

    # assert that out_of_range dates were removed from x as well
    assert data._micro_lst[3] = data._x.loc['4',:].shape[0] + 1
 

def test_indirect_find_closest_date_missing_date():
    # find missing date in svi
    # get data with proper delay to cause correction
    # assert that this date was really changed 1 back
    pass


def test_find_closest_date_out_of_range():
    # create data object
    # function(date out of range)
    # assert returns False
    pass


def test_indirect_join_x_y():
    # self.delay_table.loc[:,'micro']==self._x
    # self.delay_table.loc[:,'svi']==self._y
    pass
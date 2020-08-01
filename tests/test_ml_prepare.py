from ml_prepare import ML_prepare

import pytest

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
    assert isinstance(self._x, pd.DataFrame)
    assert isinstance(self._y, pd.DataFrame)
    assert isinstance(self.delay_table, pd.DataFrame)


def test_indirect_read_and_index_svi_tables():
    delay = 2
    data = ML_prepare(delay)
    for i in range(4):
        assert isinstance(data._svi_lst[i].index, pd.DatetimeIndex)
        assert isinstance(data._svi_lst[i].index, pd.DatetimeIndex)


def test_get_partial_table_wrong_input():
    # section = 'blah'
    # function raises AssertionError
    pass


def test_get_partial_table_no_nans():
    # get data
    # run on all sections
    # assert no rows with nans
    pass


def test_get_partial_table_x_sections_lengths():
    section_lengths = [1, 2, 3, 4]
    section_names = [1, 2, 3, 4]
    # for i in range(4):
        # table = function
        # assert table['x'].shape[1]==section_lengths[i]
    pass


def test_get_partial_table_y_labels():
    # get data
    # table = function (lables=True)
    # # assert table['y'].columns.to_list() == ['SV_label','SVI_label']
    # table = function (lables=False)
    # # assert table['y'].columns.to_list() == ['Settling_velocity','SVI']
    pass



def test_indirect_create_x_y_delayed_length():
    # get data
    # assert len(self._x) == len(self._y)
    pass


def test_indirect_create_x_y_bioreactor():
    # get data with delay = 20?
    # assert that for specific bio-reactor, the first date is indeed 20 days after
    # assert that for specific bio-reactor, the length is indeed shortened for latest date in range
    pass
 

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


# class ml_prepare, test creating delayed x y    
    # dif = timedelta(days = delay)
    # for i in range(1,5):
    #     xi = x.loc[f'{i}']
    #     yi = y.loc[f'{i}']
    #     assert xi.shape[0]==yi.shape[0]
    #     for row_i in range(len(xi.index)):
    #         assert yi.index[row_i] == xi.index[row_i] + dif
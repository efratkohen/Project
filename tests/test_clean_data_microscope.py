from clean_data_microscope import *
from files_process_save import *

import pytest
import datetime

def test_dates_to_datetime_objects_list_changed():
    micro_df_list = micro_data_read_and_split()
    assert isinstance(micro_df_list[0].loc[0,'date'], str)

    micro_df_list = dates_to_datetime_objects(micro_df_list)
    for i in range(4):
        assert isinstance(micro_df_list[i].loc[0,'date'], datetime.datetime)


def test_remove_nan_rows_index_change():
    micro_df_list = micro_data_read_and_split()
    micro_df_list = dates_to_datetime_objects(micro_df_list)
    df0 = micro_df_list[0]
    orig_len = df0.shape[0]
    # test
    remove_nan_rows(df0)
    assert df0.shape[0] > orig_len
    assert isinstance(mic0.index, pd.RangeIndex)

def test_fix_col_to_float():
    # generate_mock_df with two columns '1,000.00'
    micro_df_list = micro_data_read_and_split()
    micro_df_list = dates_to_datetime_objects(micro_df_list)
    
    df0 = micro_df_list[0]
    remove_nan_rows(df0)

    df0.iloc[1,2]='1,000.00'
    df0.iloc[8,2]='15,000.00'

    assert df0.iloc[:,2].dtype != np.float64

    # make sure it changes type
    fix_col_to_float(df0, col_i=2)
    assert df0.iloc[:,2].dtype == np.float64


def test_fix_object_cols_to_float():
    # generate_mock_df with one columns '1,000.00'
    # make sure all types are float
    pass

def test_remove_negatives():
    # generate_mock_df with negatives
    # make sure they are now is_null
    pass

def test_filaments_zero_to_nan():
    # load data
    # assert two kinds of rows are now all null
    pass

def test_assert_totals_correct():
    # load data
    # change one of the values
    # with pytest.raises(AssertionError):
        # assert_totals_correct(changed_data)
    pass

def test_clean_micro_df_changed():
    # load data
    # do function
    # check it is changed inplace
    pass

def test_clean_micro_df_list_changed():
    # load list
    # do function
    # check all dfs changed
    pass
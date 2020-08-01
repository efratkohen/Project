from clean_data_microscope import *
from files_process_save import *

import pytest

def test_dates_to_datetime_objects_list_changed():
    pass

def test_remove_nan_rows_index_change():
    # assert length 
    # assert index reset
    pass

def test_fix_col_to_float():
    # generate_mock_df with two columns '1,000.00'
    # make sure it changed type
    pass

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
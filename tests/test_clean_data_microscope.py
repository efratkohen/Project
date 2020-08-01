from clean_data_microscope import *
from files_process_save import *

import pytest
import datetime
import pandas as pd 

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
    assert df0.shape[0] < orig_len
    assert isinstance(df0.index, pd.RangeIndex)

def test_fix_col_to_float():
    # generate_mock_df with two columns '1,000.00'
    df0 = pd.DataFrame(np.random.rand(10,10))
    df0.iloc[1,2]='1,000.00'
    df0.iloc[8,2]='15,000.00'

    assert df0.iloc[:,2].dtype != np.float64

    # make sure it changes type
    fix_col_to_float(df0, col_i=2)
    assert df0.iloc[:,2].dtype == np.float64


def test_fix_object_cols_to_float():
    # generate_mock_df with two columns '1,000.00'
    df0 = pd.DataFrame(np.random.rand(10,10))
    df0.iloc[1,2]='1,000.00'
    df0.iloc[1,8]='15,000.00'
    assert not (df0.dtypes==np.float64).all()

    fix_object_cols_to_float(df0)
    assert (df0.dtypes==np.float64).all()

def test_remove_negatives():
    # generate_mock_df with negatives
    df0 = pd.DataFrame(np.random.rand(10,10))
    df0.iloc[4,4] = -1
    df0.iloc[4,8] = 0
    remove_negatives(df0)
    assert pd.isnull(df0.iloc[4,4])
    assert df0.iloc[4,8] == 0


def test_filaments_zero_to_nan():
    # load data
    micro_df_list = micro_data_read_and_split()
    micro_df_list = dates_to_datetime_objects(micro_df_list)
    micro_df = micro_df_list[1]
    remove_nan_rows(micro_df)
    fix_object_cols_to_float(micro_df)
    remove_negatives(micro_df)
    
    # create rows that are nan or zero in all filaments
    micro_df.loc[1,'Total Count- Filaments']=0
    micro_df.loc[1,'Filaments_Nocardia_index':]=np.nan

    micro_df.loc[2,'Total Count- Filaments']=8
    micro_df.loc[2,'Filaments_Nocardia_index':]=0
    
    # assert all nan
    filaments_zero_to_nan(micro_df)
    assert (pd.isnull(micro_df.loc[1, 'Total Count- Filaments':])).all()
    assert (pd.isnull(micro_df.loc[2, 'Total Count- Filaments':])).all()

def test_assert_totals_correct():
    # load data
    micro_df_list = micro_data_read_and_split()
    micro_df_list = dates_to_datetime_objects(micro_df_list)
    micro_df = micro_df_list[2]
    remove_nan_rows(micro_df)
    fix_object_cols_to_float(micro_df)
    remove_negatives(micro_df)
    filaments_zero_to_nan(micro_df)

    # change values to wrong sum
    micro_df.loc[4,'Total Count- Filaments']+=1

    with pytest.raises(AssertionError):
        assert_totals_correct(micro_df)

def test_clean_micro_df_changed():
    # load df
    micro_df_list = micro_data_read_and_split()
    micro_df_list = dates_to_datetime_objects(micro_df_list)
    before = micro_df_list[3].copy()

    # clean and check it changed
    after = clean_micro_df(micro_df_list[3])
    assert not after.equals(before)
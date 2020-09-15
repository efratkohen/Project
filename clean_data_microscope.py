import pandas as pd 
import numpy as np


def dates_to_datetime_objects(df_list: list):
    """
    Change 'date' column from string to datetime objects, in every df in the list.

    Parameters
    ----------
    df_list: list of df with column 'date'

    Return
    ----------
    df_list: list of df
    """
    for df in df_list:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.normalize()
    return df_list


def remove_nan_rows(micro_df: pd.DataFrame):
    """
    Remove rows that contain only nan values (except date column).
    Change df inplace.
    
    Parameters
    ----------
    micro_df: pd.DataFrame
   
    """
    data_cols = micro_df.columns.tolist()[1:]
    micro_df.dropna(how = 'all', subset = data_cols, inplace=True)
    micro_df.reset_index(inplace=True, drop=True)


def fix_col_to_float(micro_df: pd.DataFrame, col_i: int):
    """ 
    Fix string values with commas to float values, in column number 'col_i'.
    Change df inplace.
    
    Parameters
    ----------
    micro_df: pd.DataFrame
    col_i: int
        column index

    """
    for row_i in range(micro_df.shape[0]):
        datum = micro_df.iloc[row_i, col_i]
        if type(datum) is str and ',' in datum:
            num = datum.split(',')
            micro_df.iloc[row_i, col_i] = num[0]+num[1]

    col_name = micro_df.columns[col_i]
    micro_df.loc[:, col_name] = pd.to_numeric(micro_df[col_name])


def fix_object_cols_to_float(micro_df: pd.DataFrame):
    """
    Convert 'object' columns with string numbers to dtype float
    Change df inplace.

    Parameters
    ----------
    micro_df: pd.DataFrame
    """
    obj_cols_is = [] 
    for col_i in range(1, len(micro_df.dtypes)): # exclude 'date' column
        if micro_df.dtypes[col_i]==object:
            obj_cols_is.append(col_i)
    
    for col_i in obj_cols_is:
        fix_col_to_float(micro_df, col_i)


def remove_negatives(micro_df: pd.DataFrame):
    """
    Replaces negative values with NaN.
    Change df inplace.

    Parameters
    ----------
    micro_df: pd.DataFrame
    """

    numeric = micro_df._get_numeric_data()
    numeric.where(numeric>=0, np.nan, inplace=True)


def filaments_zero_to_nan(micro_df: pd.DataFrame):
    """
    If a row has all its "filament" columns 0 or NaN,
    turns all the "filament" values, including the "Total count- Filaments" to NaN.
    Change df inplace.

    Parameters
    ----------
    micro_df: pd.DataFrame
    """
    ## find col index of first filament:
    for i in range(len(micro_df.columns)):
        if 'Filaments' in micro_df.columns[i]:
            first_filament = i
            break

    for i in range(micro_df.shape[0]):
        # if all fillaments are NaN or Zero, turn them all, including "Total" to NaN
        if (pd.isnull(micro_df.iloc[i, first_filament + 1:])).all() or (micro_df.iloc[i, first_filament + 1:]==0).all():
            micro_df.iloc[i, first_filament:] = np.nan


def assert_totals_correct(micro_df: pd.DataFrame):
    """
    Asserts that the total count of every organism group is correct in all rows. 
    Otherwise, throughs error massage.

    Parameters
    ----------
    micro_df: pd.DataFrame
    """

    totals_dict = {'Total Count- amoeba':['ameoba_arcella','ameoba_nude ameba'], 
        'Total Count- Crawling Ciliates':['crawling ciliates_aspidisca', 'crawling ciliates_trachelopylum'], 
        'Total Count- Free swimming Ciliates':['free swimming ciliates_lionutus','free swimming ciliates_paramecium'], 
        'Total Count- Stalked Ciliates':['stalked ciliate_epistylis', 'stalked ciliate_vorticella', 'stalked ciliate_carchecium', 'stalked ciliate_tokophyra', 'stalked ciliate_podophyra', 'stalked ciliate_opercularia'], 
        'Total Count- Rotifers':['rotifer_rotifer'], 
        'Total Count- Worms':['worms_nematode', 'worms_worms'], 
        'Total Count- Spirochaetes':['spirochaetes_spirochaetes'], 
        'Total Count- Flagellats':['flagellates_peranema trich', 'flagellates_micro flagellates'], 
        'Total Count- Filaments':['Filaments_Nocardia_index','Filaments_Microthrix_index', 'Filaments_N. Limicola_index', 'Filaments_Thiothrix_index', 'Filaments_0041/0675_index', 'Filaments_0092_index', 'Filaments_1851_index', 'Filaments_beggiatoa_index', 'Filaments_zoogloea_index']
        }
    for i in range(micro_df.shape[0]):
        for group in totals_dict:
            written_sum = micro_df.loc[i, group]
            if not pd.isnull(written_sum):
                our_sum = np.sum(micro_df.loc[i, totals_dict[group]])
                assert our_sum == written_sum, (f'wrong total of group "{group}" row {i},'+ 
                                                f'sum written {written_sum}, our_sum {our_sum}')


def clean_micro_df(micro_df: pd.DataFrame):
    """
    Cleans values of microscopic dataframes with all the cleansing functions

    Parameters
    ----------
    micro_df: pd.DataFrame

    Return
    ----------
    micro_df: pd.DataFrame
    """
    remove_nan_rows(micro_df)
    fix_object_cols_to_float(micro_df)
    remove_negatives(micro_df)
    filaments_zero_to_nan(micro_df)
    assert_totals_correct(micro_df)
    return micro_df


def clean_micro_df_list(micro_df_list: list):
    """
    Loop over the 4 dataframes in the dataframe list
    and use the clean_micro_df to clean values.
    Changes all df in list inplace.

    Parameters
    ----------
    micro_df_list: list
    """
    for i in range(4):
        micro_df_list[i] = clean_micro_df(micro_df_list[i])
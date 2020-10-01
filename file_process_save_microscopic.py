import pathlib
import pandas as pd
import numpy as np
from typing import Union
#import matplotlib.pyplot as plt
#import seaborn as sns


def check_file(data_fname: Union[pathlib.Path, str]):
    """
    Check for valid file name
    accept strings and pathlib.Path objects

    Parameters
    ----------
    data_fname: pathlib.Path or str

    return
    ----------
    fname: pathlib.Path
    """

    try:
        fname = pathlib.Path(data_fname)
    except TypeError:
        print("ERROR: Please supply a string or a pathlib.Path instance.")
        raise
    if not fname.exists():
        raise ValueError(f"File {str(fname)} doesn't exist.")
    return fname

def dates_to_datetime_objects(micro_df: pd.DataFrame):
    """
    Change 'date' column from string to datetime objects and normalize the time
    Set the time to be the index of the df

    Parameters
    ----------
    df_list: list of df with column 'date'

    Return
    ----------
    df_list: list of df
    """
    
    micro_df['Time'] = pd.to_datetime(micro_df['Time'], dayfirst=True).dt.normalize()
    micro_df = micro_df.set_index('Time')
    micro_df = micro_df[~micro_df.index.duplicated(keep='first')]
    micro_df = micro_df.reset_index()
    return micro_df

def split_microscopic_to_reactor(data_micro: pd.DataFrame):
    """
    Splits the microscopic data to 4 reactors dfs and saves it in df list
    Changes the columns names to be identical in the microscopic data frame of each reactor.
    
    Parameters
    ----------
    data_micro: pd.DataFrame
    
    Returns
    -------
    micro_df_list: List of 4 dfs, each representing a bio_reactor
    """
    micro_df_list = []
    for i in range(0, 4):
        # 27 columns for each reactor, starting with 1:27...
        first_col = 1 + 27 * i
        last_col = 1 + 27 * (i + 1)
        micro_reactor_df = data_micro.iloc[:, np.r_[0, first_col:last_col]]
        micro_reactor_df.columns = [
            "date",
            "ameoba_arcella",
            "ameoba_nude ameba",
            "crawling ciliates_aspidisca",
            "crawling ciliates_trachelopylum",
            "free swimming ciliates_lionutus",
            "free swimming ciliates_paramecium",
            "stalked ciliate_epistylis",
            "stalked ciliate_vorticella",
            "stalked ciliate_carchecium",
            "stalked ciliate_tokophyra",
            "stalked ciliate_podophyra",
            "stalked ciliate_opercularia",
            "rotifer_rotifer",
            "worms_nematode",
            "worms_worms",
            "flagellates_peranema trich",
            "flagellates_micro flagellates",
            "spirochaetes_spirochaetes",
            "Filaments_Nocardia_index",
            "Filaments_Microthrix_index",
            "Filaments_N. Limicola_index",
            "Filaments_Thiothrix_index",
            "Filaments_0041_0675_index",
            "Filaments_0092_index",
            "Filaments_1851_index",
            "Filaments_beggiatoa_index",
            "Filaments_zoogloea_index",
        ]
        micro_df_list.append(micro_reactor_df)

    return micro_df_list

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

def save_dfs_to_csv(df_list: list, data_name: str):
    """
    Save the split, cleaned list of 4 bio reactors dataframes to csv file.
    If files already exists, skips saving.

    Parameters
    ----------
    df_list: list
    data_name: str
        desirable csv file name
    """
    assert data_name in {"svi", "micro"}, 'data_name invalid, expected "svi"/"micro"'
    for i in range(4):
        fname = pathlib.Path("clean_tables/" + f"{data_name}_{i+1}.csv")
        if not pathlib.Path(fname).is_file():  # only if it does not exist yet
            df_list[i].to_csv(fname, index=False)


if __name__ == "__main__":
    ##### process micro data ######
    micro_path = check_file("micro - total.csv")
    micro_df = pd.read_csv(micro_path)
    micro_df = dates_to_datetime_objects(micro_df)
    micro_df_list = split_microscopic_to_reactor(micro_df)
    clean_micro_df_list(micro_df_list)
    save_dfs_to_csv(micro_df_list,"micro")

   

   



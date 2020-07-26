from typing import Union, Tuple
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def check_file(data_fname: Union[pathlib.Path, str]):
    """Check for valid file
    accept strings and pathlib.Path objects"""
    try:
        fname = pathlib.Path(data_fname)
    except TypeError:
        print("ERROR: Please supply a string or a pathlib.Path instance.")
        raise
    if not fname.exists():
        raise ValueError(f"File {str(fname)} doesn't exist.")
    return fname


def read_data(data_fname: Union[pathlib.Path, str]) -> pd.DataFrame:
    """Reads data into DF"""
    data = pd.read_csv(data_fname)
    return data


def convert_str_to_date(s: str) -> datetime:
    '''
    Gets str like '04/08/2017 00:00'
    Removes hours and turns it to datetime object.
    '''
    d_m_y = s.split()[0]
    date_obj = datetime.strptime(d_m_y, '%d/%m/%Y').date()
    return date_obj


def convert_dates_to_date_object(data: pd.DataFrame):
    '''
    Runs on all rows of df. 
    turns strings of 'date' columns to date objects.
    '''
    for i in range(data.shape[0]):
        data.iloc[i, 0] = convert_str_to_date(data.iloc[i, 0])


def check_SVI_values(data: pd.DataFrame) -> pd.DataFrame:
    """check and replace incorrect values with nan"""

    data.replace(0, np.nan, inplace=True)
    ranges_lst = [(0, 6), (0, 1000), (500, 4000)]
    col_names_lst = ["SV", "volume", "mlss"]

    for i in range(3):
        name = col_names_lst[i]
        low = ranges_lst[i][0]
        high = ranges_lst[i][1]
        data.loc[:, data.columns.str.contains(name)] = data.loc[
            :, data.columns.str.contains(name)
        ].apply(lambda x: [y if low < y < high else np.nan for y in x])

    return data


def SVI_calculate(data_svi: pd.DataFrame) -> pd.DataFrame:
    """Add column of SVI caculation fot each reactor"""
    for i in range(1, 5):
        data_svi[f"SVI{i}"] = data_svi[f"volume reactor {i}"] * 1000 / data_svi[f"mlss reactor {i}"]

    return data_svi


def split_SVI_to_reactor(data_svi: pd.DataFrame):
    """Split the SVI data frame to 4 dataframes for each reactor.
    Change the columns names to be identical
    
    Returns
    -------
    svi_df_lst: List of 4 dfs, each representing a bio_reactor
    """
    svi_df_lst = []
    for i in range(1, 5):
        reactor_df = data_svi[["Date", f"SV reactor {i}", f"SVI{i}"]]
        reactor_df.columns = ["date", "Settling_velocity", "SVI"]
        svi_df_lst.append(reactor_df)
    return svi_df_lst


def label_data(data_svi: pd.DataFrame, label_SVI: list, label_SV: list) -> pd.DataFrame:
    """add labels column for SVI and SV results as bad, reasonable or good"""

    data_svi["SV_label"] = np.where(
        data_svi.loc[:, "Settling_velocity"] <= label_SV[0],
        "bad",
        np.where(data_svi.loc[:, "Settling_velocity"] <= label_SV[1], "reasonable", "good"),
    )
    data_svi["SVI_label"] = np.where(
        data_svi.loc[:, "SVI"] >= label_SVI[0],
        "bad",
        np.where(data_svi.loc[:, "SVI"] >= label_SVI[1], "reasonable", "good"),
    )
    return data_svi


def split_microscopic_to_reactor(data_micro: pd.DataFrame):
    """Split the microscopic data frame to 4 dataframes for each reactor.
    Change the columns names to be identical
    
    Returns
    -------
    micro_df_list: List of 4 dfs, each representing a bio_reactor
    """
    micro_df_list = []
    for i in range(0,4):
        # 37 columns for each reactor, starting with 1:38...
        first_col = 1+36*i ## fix
        last_col = 1+36*(i+1) ## fix
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
            "flagellates_peranema trich",
            "flagellates_micro flagellates",
            "worms_worms",
            "spirochaetes_spirochaetes",
            "Total Count- amoeba",
            "Total Count- Crawling Ciliates",
            "Total Count- Free swimming Ciliates",
            "Total Count- Stalked Ciliates",
            "Total Count- Rotifers",
            "Total Count- Worms",
            "Total Count- Spirochaetes",
            "Total Count- Flagellats",
            # "Total Count- Free Bacteria",###
            "Total Count- Filaments",
            "Filaments_Nocardia_index",
            "Filaments_Microthrix_index",
            "Filaments_N. Limicola_index",
            "Filaments_Thiothrix_index",
            "Filaments_0041/0675_index",
            "Filaments_0092_index",
            "Filaments_1851_index",
            "Filaments_beggiatoa_index",
            "Filaments_zoogloea_index",
        ]
        micro_df_list.append(micro_reactor_df)

    return micro_df_list



def remove_nan_rows(micro_df: pd.DataFrame):
    '''
    Removes rows that contain only nan values (except date column)

    Returns
    ------
    changes inplace
    '''
    data_cols = micro_df.columns.tolist()[1:]
    micro_df.dropna(how = 'all', subset = data_cols, inplace=True)
    micro_df = micro_df.reset_index(drop=True)
    return micro_df


def fix_spiro_and_float(micro_df: pd.DataFrame):
    
    # fix values with commas
    for i in range(micro_df.shape[0]):
        datum = micro_df.loc[i,'spirochaetes_spirochaetes']
        if type(datum) is not str:
            print(f'row {i} value {datum}')
        elif ',' in datum:
            num = datum.split(',')
            micro_df.loc[i,'spirochaetes_spirochaetes'] = num[0]+num[1]

    micro_df["spirochaetes_spirochaetes"] = pd.to_numeric(micro_df["spirochaetes_spirochaetes"])


def remove_negatives(micro_df: pd.DataFrame):
    '''
    Replaces negative values with NaN.

    Returns
    ------
    changes inplace
    '''
    numeric = micro_df._get_numeric_data()
    numeric[numeric < 0] == np.nan


def filaments_zero_to_nan(micro_df: pd.DataFrame):
    '''
    If a row has all its "filament" columns 0 - 
    turns all the "filament" values to NaN.

    Returns
    ------
    changes inplace
    '''
    ## find col index of first filament:
    for i in range(len(micro_df.columns)):
        if 'Filaments' in micro_df.columns[i]:
            first_filament = i
            break

    for i in range(micro_df.shape[0]):
        if (micro_df.iloc[i, first_filament:]==0).all():
            micro_df.iloc[i, first_filament:]= np.nan
            print(f'row {i} was fixed to nan in micro_data')


def assert_totals_correct(micro_df: pd.DataFrame):
    '''
    Asserts that the total count of every kind of every row
    is correct. 
    Otherwise, through error massage.
    '''
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
        # print(f'i = {i}')
        for group in totals_dict:
            # print(f'group {group}')
            written_sum = micro_df.loc[i, group]
            if not pd.isnull(written_sum):
                our_sum = np.sum(micro_df.loc[i, totals_dict[group]])
                assert our_sum == written_sum, (f'wrong total of group "{group}" row {i},'+ 
                                                f'sum written {written_sum}, our_sum {our_sum}')
    
def clean_micro_df(micro_df: pd.DataFrame):
    '''
    ...
    '''
    micro_df = remove_nan_rows(micro_df)
    fix_spiro_and_float(micro_df)
    remove_negatives(micro_df)
    filaments_zero_to_nan(micro_df)
    assert_totals_correct(micro_df)
    return micro_df






    


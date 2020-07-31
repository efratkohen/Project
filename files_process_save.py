import project as pr
import pathlib
import pandas as pd
import numpy as np
from typing import Union # , Tuple later
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


def micro_data_read_and_split(micro_fname: str ='microscopic_data.csv'):
    '''
    Reads and splits.

    Returns
    -------
    - micro_df_list: List of 4 dfs, each representing a bio_reactor
    '''
    micro_path = check_file(micro_fname)
    data_microscopic = read_data(micro_path)
    micro_df_list = split_microscopic_to_reactor(data_microscopic)
    return micro_df_list

def split_microscopic_to_reactor(data_micro: pd.DataFrame):
    """Split the microscopic data frame to 4 dataframes for each reactor.
    Change the columns names to be identical
    
    Returns
    -------
    micro_df_list: List of 4 dfs, each representing a bio_reactor
    """
    micro_df_list = []
    for i in range(0,4):
        # 36 columns for each reactor, starting with 1:37...
        first_col = 1+36*i 
        last_col = 1+36*(i+1) 
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


def dates_to_datetime_objects(df_list: list):
    '''
    '''
    for df in df_list:
        df['date'] = pd.to_datetime(df['date'])
    return df_list


def svi_data_read_calculate_and_split(svi_fname:str="SVI.csv"):
    '''
    Reads, computes values SVI, and splits.

    Returns
    -------
    - svi_df_list: List of 4 dfs, each representing a bio_reactor
    '''
    svi_path = check_file(svi_fname)
    data_svi = read_data(svi_path)
    data_svi = check_svi_values_range(data_svi)
    data_svi_computed = svi_calculate(data_svi)
    svi_df_list = split_svi_to_reactor(data_svi_computed)
    return svi_df_list


def check_svi_values_range(svi_data: pd.DataFrame) -> pd.DataFrame:
    '''
    check and replace incorrect values with nan
    '''

    svi_data.replace(0, np.nan, inplace=True)
    ranges_lst = [(0, 6), (0, 1000), (500, 4000)]
    col_names_lst = ["SV", "volume", "mlss"]

    for i in range(3):
        name = col_names_lst[i]
        low = ranges_lst[i][0]
        high = ranges_lst[i][1]
        svi_data.loc[:, svi_data.columns.str.contains(name)] = svi_data.loc[
            :, svi_data.columns.str.contains(name)
        ].apply(lambda x: [value if low < value < high else np.nan for value in x])

    return svi_data


def svi_calculate(data_svi: pd.DataFrame) -> pd.DataFrame:
    """Add column of SVI caculation fot each reactor"""
    for i in range(1, 5):
        data_svi.loc[:,f"SVI{i}"] = data_svi[f"volume reactor {i}"] * 1000 / data_svi[f"mlss reactor {i}"]

    return data_svi


def split_svi_to_reactor(data_svi: pd.DataFrame):
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


def set_datetime_index(df_list: list):
    for i in range(4):
        df_list[i].set_index('date', inplace=True)
        df_list[i].index = pd.to_datetime(df_list[i].index)

def interpolate_svi_dfs(svi_df_list: list):
    for i in range(4):
        svi_df_list[i].interpolate(inplace=True, method='time')

def svi_label(svi_df_list: list):
    '''
    '''
    # define borders between bad / reasonable / good results
    SVI_label=[160.0, 120.0]
    SV_label=[3.0, 3.5] 
    for svi_df in svi_df_list:
        svi_df = label_data(svi_df, SVI_label, SV_label)

def label_data(data_svi: pd.DataFrame, label_SVI: list, label_SV: list) -> pd.DataFrame:
    """add labels column for SVI and SV results as bad, reasonable or good"""

    data_svi.loc[:,"SV_label"] = np.where(
        data_svi.loc[:, "Settling_velocity"] <= label_SV[0],
        "bad",
        np.where(data_svi.loc[:, "Settling_velocity"] <= label_SV[1], "reasonable", "good"),
    )
    data_svi.loc[:,"SVI_label"] = np.where(
        data_svi.loc[:, "SVI"] >= label_SVI[0],
        "bad",
        np.where(data_svi.loc[:, "SVI"] >= label_SVI[1], "reasonable", "good"),
    )
    return data_svi


if __name__=='__main__':
    ##### micro data ######
    micro_df_list = micro_data_read_and_split()
    # micro_df_list = dates_to_objects(micro_df_list) # old later
    micro_df_list = dates_to_datetime_objects(micro_df_list)
    # pr.clean_micro_df_list(micro_df_list)
    
    # # save to csv
    # for i in range(4):
    #     fname = pathlib.Path('clean_tables/'+f'micro_{i}.csv')
    #     if not pathlib.Path(fname).is_file(): # only if it does not exist yet
    #         micro_df_list[i].to_csv(fname, index=False)

    # ##### SVI data ######
    svi_df_list = svi_data_read_calculate_and_split()
    set_datetime_index(svi_df_list)
    interpolate_svi_dfs(svi_df_list)
    svi_label(svi_df_list)

    # # save to csv
    # for i in range(4):
    #     fname = pathlib.Path('clean_tables/'+f'svi_{i}.csv')
    #     if not pathlib.Path(fname).is_file(): # only if it does not exist yet
    #         svi_df_list[i].to_csv(fname)

    
    



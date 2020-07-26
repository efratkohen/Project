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

    data = data.replace(0, np.nan)
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
    data frame
        data_reactor1, data_reactor2, data_reactor3, data_reactor4
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
    data frame
        data_reactor1, data_reactor2, data_reactor3, data_reactor4
    """
    micro_df_list = []
    for i in range(0,4):
        # 37 columns for each reactor, starting with 1:38...
        first_col = 1+37*i
        last_col = 1+37*(i+1)
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
            "Total Count- Free Bacteria",
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

# for i in range(0,4):
#     a = 1+37*i
#     b = 1+37*(i+1)
#     print(a,b)

# def join microscopic and SVI data (data: pd.DataFrame, data_reactor1: pd.DataFrame, data_reactor2: pd.DataFrame, data_reactor3: pd.DataFrame, data_reactor4: pd.DataFrame,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     """

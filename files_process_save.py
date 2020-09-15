import clean_data_microscope as cdm
import clean_data_svi as cds
import pathlib
import pandas as pd
import numpy as np
from typing import Union


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


def read_data(data_fname: Union[pathlib.Path, str]) -> pd.DataFrame:
    """
    Reads data into DF
    
    Parameters
    ----------
    data_fname: pathlib.Path or str

    return
    ----------
    data: pd.DataFrame
    """
    data = pd.read_csv(data_fname)
    return data


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
        # 36 columns for each reactor, starting with 1:37...
        first_col = 1 + 36 * i
        last_col = 1 + 36 * (i + 1)
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


def micro_data_read_and_split(micro_fname: str = "microscopic_data.csv"):
    """
    Reads csv file and splits the data to 4 reactors df list.
    Combine the functions check_file, read_data and split_microscopic_to_reactor to one function.
    
    Parameters
    ----------
    micro_fname_fname: str

    Returns
    -------
    micro_df_list: List of 4 dfs, each representing a bio_reactor
    """
    micro_path = check_file(micro_fname)
    data_microscopic = read_data(micro_path)
    micro_df_list = split_microscopic_to_reactor(data_microscopic)
    return micro_df_list


def svi_data_read_calculate_and_split(svi_fname: str = "SVI.csv"):
    """
    Reads CSV file, computes values SVI, and splits to 4 bio reactors.

    Parameters
    ----------
    svi_fname: str

    Returns
    -------
    svi_df_list: List of 4 dfs, each representing a bio_reactor
    """
    svi_path = check_file(svi_fname)
    data_svi = read_data(svi_path)
    data_svi = cds.check_svi_values_range(data_svi)
    data_svi_computed = cds.svi_calculate(data_svi)
    svi_df_list = split_svi_to_reactor(data_svi_computed)
    return svi_df_list


def split_svi_to_reactor(data_svi: pd.DataFrame):
    """Split the SVI data frame to 4 dataframes for each reactor.
    Change the columns names to be identical.

    Parameters
    ----------
    data_svi: pd.DataFrame
    
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
        fname = pathlib.Path("clean_tables/" + f"{data_name}_{i}.csv")
        if not pathlib.Path(fname).is_file():  # only if it does not exist yet
            if data_name=="micro":
                df_list[i].to_csv(fname, index=False)
            else:
                df_list[i].to_csv(fname)


if __name__ == "__main__":
    ##### process micro data ######
    micro_df_list = micro_data_read_and_split()
    micro_df_list = cdm.dates_to_datetime_objects(micro_df_list)
    cdm.clean_micro_df_list(micro_df_list)

    # save to csv
    save_dfs_to_csv(micro_df_list, "micro")

    ##### process SVI data ######
    svi_df_list = svi_data_read_calculate_and_split()
    cds.set_datetime_index(svi_df_list)
    cds.interpolate_svi_dfs(svi_df_list)
    cds.svi_label(svi_df_list)

    # save to csv
    save_dfs_to_csv(svi_df_list, "svi")


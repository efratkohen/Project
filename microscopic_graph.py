import pathlib
import pandas as pd
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
import clean_data_microscope as cdm
import files_process_save as fps

def dates_to_datetime_objects(micro_df: pd.DataFrame):
    """Change 'date' column from string to datetime objects and normalize the time
    Set the time to be the index of the df"""
    
    micro_df['Time'] = pd.to_datetime(micro_df['Time'], dayfirst=True).dt.normalize()
    micro_df = micro_df.set_index('Time')
    micro_df = micro_df[~micro_df.index.duplicated(keep='first')]
    return micro_df

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
    cdm.fix_object_cols_to_float(micro_df)
    cdm.remove_negatives(micro_df)
    return micro_df

def create_tidy_df_for_all_microorganisms(micro_df: pd.DataFrame, organisms_ist: list) -> list:
    """create 'tidy' dataframes for all microorganisms
    add month and season columns
    sort the df by the reactor column
    return df_list"""

    micro_df_list = []
    for i in range (0, 27):
        micro_org_df = micro_df.iloc[:, np.r_[i, i+27, i+27*2, i+27*3]]
        micro_org_df = micro_org_df.stack().reset_index()
        micro_org_df.columns = ['Time', 'Reactor', organisms_ist[i]]
        for j in range(1,5):
            micro_org_df = micro_org_df.replace(f"{organisms_ist[i]}{j}", f"Reactor {j}")
        micro_org_df = micro_org_df.sort_values(by=['Reactor'])
        micro_org_df['Month'] = pd.DatetimeIndex(micro_org_df['Time']).month
        micro_org_df['Year'] = pd.DatetimeIndex(micro_org_df['Time']).year
        micro_org_df['Season'] = micro_org_df.Month.apply(lambda x: 'spring' if x >= 3 and x<=5 else(
                                                   'summer' if x >= 6 and x<=8 else( 
                                                   'fall' if x >= 9 and x<=11 else('winter'))))
        micro_df_list.append(micro_org_df)
    return micro_df_list

def plot_graph_of_microorganisms_by_time(micro_df_list: list, organisms_ist: list):
    """plot scatter graph of all microorganisms by time
    each reactor result will be in diferent color
    save the images in "figures/microscopic_organisms" file
    """

    for i in range (0, 27):
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        g = sns.scatterplot(data=micro_df_list[i], x="Time", y=organisms_ist[i], hue="Reactor", ax=ax)
        plt.xlim(micro_df_list[i]['Time'].min(), micro_df_list[i]['Time'].max())
        plt.savefig(f"figures/microscopic_organisms/{organisms_ist[i]}.png", dpi=100, bbox_inches="tight")
        plt.close()

def plot_graph_of_microorganisms_by_month(micro_df_list: list, organisms_ist: list):
    """plot scatter graph of all microorganisms by month
    save the images in "figures\microorganisms_month" file
    """

    for i in range (0, 27):
        plt.rcParams.update({'figure.max_open_warning': 0})
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        g = sns.scatterplot(data=micro_df_list[i], x="Month", y=organisms_ist[i], ax=ax)
        plt.xticks(np.arange(0, 13, step=1))
        plt.savefig(f"figures/microorganisms_month/{organisms_ist[i]}.png", dpi=100, bbox_inches="tight")
        plt.close()

def plot_graph_of_microorganisms_by_month_year(micro_df_list: list, organisms_ist: list):
    """plot heatmap graph of all microorganisms by month and year
    save the images in "figures\microorganisms_month_year" file
    """

    for i in range (0, 27):
        plt.rcParams.update({'figure.max_open_warning': 0})
        table = pd.pivot_table(micro_df_list[i], values=organisms_ist[i], index=['Month'], columns=['Year'])
        ax = sns.heatmap(table, cmap="rocket_r")
        plt.title(organisms_ist[i])
        plt.xticks(rotation=90)
        plt.savefig(f"figures/microorganisms_month_year/{organisms_ist[i]}.png", dpi=100, bbox_inches="tight")
        plt.close()
    

if __name__ == "__main__":
    """ Creatr and save image graph for microorganisms in each reactor by time, month and year 
    For using this code the CSV file of microorganisms 
    need to be with columns name like in organisms_ist + a number of the reactor in the end
    """
    organisms_ist= [
            "arcella",
            "nude ameba",
            "aspidisca",
            "trachelopylum",
            "lionutus",
            "paramecium",
            "epistylis",
            "vorticella",
            "carchecium",
            "tokophyra",
            "podophyra",
            "opercularia",
            "rotifer",
            "nematode",
            "worms",
            "peranema trich",
            "micro flagellates",
            "spirochaetes",
            "Nocardia",
            "Microthrix",
            "N. Limicola",
            "Thiothrix",
            "0041_0675 ",
            "0092 ",
            "1851 ",
            "beggiatoa",
            "zoogloea"
        ]
    micro_path = fps.check_file("micro - total.csv")
    data_microscopic = pd.read_csv(micro_path)
    data_microscopic = dates_to_datetime_objects(data_microscopic)
    data_microscopic = clean_micro_df(data_microscopic)
    micro_df_list = create_tidy_df_for_all_microorganisms(data_microscopic, organisms_ist)
    plot_graph_of_microorganisms_by_time(micro_df_list, organisms_ist)
    plot_graph_of_microorganisms_by_month(micro_df_list, organisms_ist)
    plot_graph_of_microorganisms_by_month_year(micro_df_list, organisms_ist)
    

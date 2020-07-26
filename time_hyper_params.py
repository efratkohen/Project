import project as pr
from datetime import datetime


def SVI_read_and_process():
    '''
    Uses helper function to process SVI.csv.
    Reads SVI data sheet.
    Fixes wrong values. 
    Computes SV and SVI.
    Splits to 4 bio reactors.
    Labels data by determined values.
    Turns strings of 'date' columns to date objects.

    Return
    -------
    - bio_reactor_df_list: List of 4 dfs, each representing a bio_reactor
    '''

    SVI_label=[190.0, 160.0]
    # define between bad, reasonable and good result for SVI
    SV_label=[2.5, 3.0] 
    # define between bad, reasonable and good result for SV

    fname = "SVI.csv"
    data_fname = pr.check_file(fname)
    data_SVI = pr.read_data(data_fname)
    data_SVI = pr.check_SVI_values(data_SVI)
    data_SVI = data_SVI.interpolate()
    data_SVI_clean = pr.SVI_calculate(data_SVI)
    data_reactor1, data_reactor2, data_reactor3, data_reactor4 = pr.split_SVI_to_reactor(data_SVI_clean)
    bio_reactor_df_list = [data_reactor1, data_reactor2, data_reactor3, data_reactor4]

    for bio_reactor_df in bio_reactor_df_list:
        bio_reactor_df = pr.label_data(bio_reactor_df, SVI_label, SV_label)
        bio_reactor_df = pr.convert_dates_to_date_object(bio_reactor_df)
    return bio_reactor_df_list

def microscopic_data_read_and_process():
    '''
    Uses helper function to process microscopic_data.csv.
    Reads microscopic data sheet.
    Splits to 4 bio reactors.
    Turns strings of 'date' columns to date objects.

    Return
    -------
    - bio_reactor_df_list: List of 4 dfs, each representing a bio_reactor
    '''

    fname = "microscopic_data.csv"
    data_fname = pr.check_file(fname)
    data_microscopic = pr.read_data(data_fname)
    data_reactor1, data_reactor2, data_reactor3, data_reactor4 = pr.split_microscopic_to_reactor(data_microscopic)
    bio_reactor_df_list = [data_reactor1, data_reactor2, data_reactor3, data_reactor4]

    for bio_reactor_df in bio_reactor_df_list:
        bio_reactor_df = pr.convert_dates_to_date_object(bio_reactor_df)
    return bio_reactor_df_list


def turn_str_to_date(date_s:str) -> datetime:
    '''
    Gets str like '04/08/2017 00:00'
    Removes hours and turns it to datetime object.
    '''
    


if __name__=='__main__':
    svi_df_list = SVI_read_and_process()
    microscopic_df_list = microscopic_data_read_and_process()
    svi1 = svi_df_list[0]
    microscopic1 = microscopic_df_list[0]
    # svi1.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_data_svi1.xlsx", index=False, header=True)
    # microscopic1.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_data_microscopic1.xlsx", index=False, header=True)
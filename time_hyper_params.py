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

    Return
    -------
    - bio_reactor_df_list: List of 4 dfs, each representing a bio_reactor
    '''

    SVI_label=[190.0, 160.0]
    # define between bad, reasonable and good result for SVI
    SV_label=[2.5, 3.0] 
    # define between bad, reasonable and good result for SV

    svi_fname = "SVI.csv"
    data_fname = pr.check_file(svi_fname)
    data_SVI = pr.read_data(data_fname)
    pr.convert_dates_to_date_object(data_SVI)
    data_SVI = pr.check_SVI_values(data_SVI)
    data_SVI = data_SVI.interpolate()
    data_SVI_clean = pr.SVI_calculate(data_SVI)

    bio_reactor_df_list = pr.split_SVI_to_reactor(data_SVI_clean)

    for bio_reactor_df in bio_reactor_df_list:
        bio_reactor_df = pr.label_data(bio_reactor_df, SVI_label, SV_label)
    return bio_reactor_df_list


def turn_str_to_date(date_s:str) -> datetime:
    '''
    Gets str like '04/08/2017 00:00'
    Removes hours and turns it to datetime object.
    '''
    pass



if __name__=='__main__':
    # svi_df_list1 = SVI_read_and_process()
    SVI_label=[190.0, 160.0]
    # define between bad, reasonable and good result for SVI
    SV_label=[2.5, 3.0] 
    # define between bad, reasonable and good result for SV

    svi_fname = "SVI.csv"
    data_fname = pr.check_file(svi_fname)
    data_SVI = pr.read_data(data_fname)
    # x = pr.read_data(data_fname)
    pr.convert_dates_to_date_object(data_SVI)

    data_SVI = pr.check_SVI_values(data_SVI)

    x = data_SVI.copy()

    data_SVI.interpolate(inplace=True)

    time_inter = data_SVI.interpolate(method='time')
    # data_SVI_clean = pr.SVI_calculate(data_SVI)

    # svi_df_list = pr.split_SVI_to_reactor(data_SVI_clean)

    # for bio_reactor_df in svi_df_list:
    #     bio_reactor_df = pr.label_data(bio_reactor_df, SVI_label, SV_label)
    
    # svi2 = svi_df_list[1]
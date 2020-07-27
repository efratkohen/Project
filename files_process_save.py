import project as pr
import pathlib
import pandas as pd

def micro_data_read():
    '''
    Reads and splits.

    Returns
    -------
    - micro_df_list: List of 4 dfs, each representing a bio_reactor
    '''

    micro_fname = "microscopic_data.csv"
    micro_path = pr.check_file(micro_fname)
    data_microscopic = pr.read_data(micro_path)
    micro_df_list = pr.split_microscopic_to_reactor(data_microscopic)
    return micro_df_list

def dates_to_objects(df_list: list):
    '''
    '''
    for df in df_list:
        df = pr.convert_dates_to_date_object(df)
    return df_list

def SVI_data_read():
    '''
    Reads, computes values SVI, and splits.

    Returns
    -------
    - svi_df_list: List of 4 dfs, each representing a bio_reactor
    '''
    svi_fname = "SVI.csv"
    svi_path = pr.check_file(svi_fname)
    data_svi = pr.read_data(svi_path)
    data_svi = pr.check_svi_values(data_svi)
    data_svi_computed = pr.svi_calculate(data_svi)
    svi_df_list = pr.split_svi_to_reactor(data_svi_computed)
    return svi_df_list

def set_datetime_index(df_list: list):
    for i in range(4):
        df_list[i].set_index('date', inplace=True)
        df_list[i].index = pd.to_datetime(df_list[i].index)

def interpolate_svi_dfs(svi_df_list: list):
    for i in range(4):
        svi_df_list[i].interpolate(inplace=True, method='time')

def svi_lable(svi_df_list: list):
    SVI_label=[190.0, 160.0]
    "define between bad, reasonable and good result for SVI"
    SV_label=[2.5, 3.0] 
    "define between bad, reasonable and good result for SV"
    for svi_df in svi_df_list:
        svi_df = pr.label_data(svi_df, SVI_label, SV_label)



if __name__=='__main__':
    ##### micro data ######
    micro_df_list = micro_data_read()
    micro_df_list = dates_to_objects(micro_df_list)
    pr.clean_micro_df_list(micro_df_list)
    
    # save to csv
    for i in range(4):
        fname = pathlib.Path('clean_tables/'+f'micro_{i}.csv')
        if not pathlib.Path(fname).is_file(): # only if it does not exist yet
            micro_df_list[i].to_csv(fname, index=False)

    ##### SVI data ######
    svi_df_list = SVI_data_read()
    svi_df_list = dates_to_objects(svi_df_list)
    set_datetime_index(svi_df_list)
    interpolate_svi_dfs(svi_df_list)
    svi_lable(svi_df_list)

    # save to csv
    for i in range(4):
        fname = pathlib.Path('clean_tables/'+f'svi_{i}.csv')
        if not pathlib.Path(fname).is_file(): # only if it does not exist yet
            svi_df_list[i].to_csv(fname)

    
    



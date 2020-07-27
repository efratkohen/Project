import project as pr
import pathlib
import pandas as pd

def micro_data_read():
    '''
    Uses helper function to process microscopic_data.csv.
    Reads microscopic data sheet.
    Splits to 4 bio reactors.
    Turns strings of 'date' columns to date objects.
    Return
    -------
    - bio_reactor_df_list: List of 4 dfs, each representing a bio_reactor
    '''

    micro_fname = "microscopic_data.csv"
    micro_path = pr.check_file(micro_fname)
    data_microscopic = pr.read_data(micro_path)
    micro_df_list = pr.split_microscopic_to_reactor(data_microscopic)
    return micro_df_list

def dates_to_objects(micro_df_list: pd.DataFrame):
    '''
    '''
    for micro_df in micro_df_list:
        micro_df = pr.convert_dates_to_date_object(micro_df)
    return micro_df_list

if __name__=='__main__':
    micro_df_list = micro_data_read()
    micro_df_list = dates_to_objects(micro_df_list)
    pr.clean_micro_df_list(micro_df_list)
    for i in range(4):
        fname = pathlib.Path('clean_tables/'+f'micro_{i}.csv')
        if not pathlib.Path(fname).is_file(): # only if it does not exist yet
            micro_df_list[i].to_csv(fname)
    
    



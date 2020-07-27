import project as pr

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
    bio_reactor_df_list = pr.split_microscopic_to_reactor(data_microscopic)

    for bio_reactor_df in bio_reactor_df_list:
        bio_reactor_df = pr.convert_dates_to_date_object(bio_reactor_df)
    return bio_reactor_df_list

if __name__=='__main__':
    micro_df_list = microscopic_data_read_and_process()
    micro1 = micro_df_list[0]
    micro2 = micro_df_list[1]
    micro3 = micro_df_list[2]
    micro4 = micro_df_list[3]



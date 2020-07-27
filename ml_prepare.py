import pandas as pd 

if __name__=='__main__':
    # read files
    micro0 = pd.read_csv('clean_tables/micro_0.csv', index_col=False)
    svi_0 = pd.read_csv('clean_tables/svi_0.csv', index_col=False)

    # set index as datetime
    


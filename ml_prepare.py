import pandas as pd 
from datetime import datetime, timedelta
import files_process_save as fps
import matplotlib.pyplot as plt 

class ml_prepare():

    def __init__(self):
        self._svi_lst = self.read_and_index_svi_tables()
        self._micro_lst = self.read_and_index_micro_tables()
        self._x = 0
        self._y = 0
        self._delay = 0
    
    @property
    def svi_lst(self):
        return self._svi_lst
    
    @property
    def micro_lst(self):
        return self._micro_lst
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y

    @property
    def delay(self):
        return self._delay

    def read_and_index_svi_tables(self):
        svi_tables = self.__read_clean_tables('svi')
        fps.set_datetime_index(svi_tables)
        return svi_tables
    
    def read_and_index_micro_tables(self):
        micro_tables = self.__read_clean_tables('micro')
        fps.set_datetime_index(micro_tables)
        return micro_tables

    def __read_clean_tables(self, data_type: str):
        assert data_type in {'micro','svi'}, '"data_type" must be "micro" / "svi"'
        clean_tables_lst = []
        for i in range(4):
            table = pd.read_csv(f'clean_tables/{data_type}_{i}.csv', index_col=False)
            clean_tables_lst.append(table)
        return clean_tables_lst

    def plot_svi(self):
        fig_svi, axes = plt.subplots(2, 1)
        fig_svi.suptitle('SV and SVI per reactors')
        for i in range(4):
            axes[0].plot(self.svi_lst[i]['SVI'], label = f'bio reactor {i+1}')
            axes[1].plot(self.svi_lst[i]['Settling_velocity'])
        axes[0].set_ylabel('SVI')
        axes[0].set_xticks([])
        axes[0].legend()
        plt.xticks(rotation=70)
        plt.show()



    def create_x_and_y(self, days_delay: int):
        '''
        Returns:
        ------
        - micro_x - df of all 4 bio-reactors one after the other
        - svi_y
        '''
        # update self._delay
        
        # create y_list
        # for each bioreactor:
            # create svi_y
            # loop on dates of micro[i].
                # function - find in svi[i] closest date of delay
                # if exists, add row to svi_y. if out of bound, remove rows from here on from this micro[i]
            # append svi_y to y_list
        
        # concat dfs: https://stackoverflow.com/questions/43897018/pandas-to-datetime-then-concat-on-datetime-index
        # with index: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
        # concat all micros to x
        # concat all y_list to y

        # update self._x, self._y, 

    def find_closest_date(self, bio_reactor_i: int, date0):
        '''
        Gets bio_reactor and date, 
        Finds in svi[bio_reactor_i] the closest row with date = date0 + delay
        If it is after the last date, return False
        If there is no such row, goes to the closest previous row.
        '''
        self._delay = 6

        delay_time = timedelta(days = self._delay)
        final_date = date0 + delay_time
        
        # if out of bounds
        last_date = self._svi_lst[bio_reactor_i].index[-1]
        if final_date > last_date:
            print('out of bounds') 
            return False

        while True:
            if final_date in self._svi_lst[bio_reactor_i].index:
                return final_date
            else:
                print(f'date {final_date} is not found in svi bio reactor {bio_reactor_i}')
                final_date -= timedelta(days=1) # go one date back


            

if __name__=='__main__':
    data = ml_prepare()
    
    
    # data.plot_svi()

svi0 = data.svi_lst[0]
micro0 = data.micro_lst[0]

# x = micro0.index[0]
# y = svi0.index[4]
# d = timedelta(days=4)
# for i in range(4):
#     count = 0
#     for date in data._micro_lst[i].index:
#         res = data.find_closest_date(bio_reactor_i=i, date0=date)
#         # print(res)
#         if res:
#             count+=1
#     print(f'bio {i} found = {count}')


    


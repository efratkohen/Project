import pandas as pd 
import files_process_save as fps
import matplotlib.pyplot as plt 

class ml_prepare():

    def __init__(self):
        self._svi_lst = self.read_and_index_svi_tables()
        self._micro_lst = self.read_and_index_micro_tables()   
    
    @property
    def svi_lst(self):
        return self._svi_lst
    
    @property
    def micro_lst(self):
        return self._micro_lst

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



if __name__=='__main__':
    data = ml_prepare()
    svi0 = data.svi_lst[0]
    
    data.plot_svi()


    


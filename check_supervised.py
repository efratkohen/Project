import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from project import *
from time_hyper_params import *
from supervised import *

if __name__=='__main__':
    svi_df_list = SVI_read_and_process()
    microscopic_df_list = microscopic_data_read_and_process()
    microscopic_df_list = drop_nan_rows(microscopic_df_list)
    microscopic1 = microscopic_df_list[0]
    #print(microscopic1.head)

    # svi1 = svi_df_list[0]
    # svi1.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_svi1.xlsx", index=False, header=True)
    # microscopic1.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_microscopic1.xlsx", index=False, header=True)
    # join_data=svi1.set_index('date', drop=False).join(microscopic1.set_index('date'))
    # join_data.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_re1.xlsx", index=False, header=True)
    # X = join_data.iloc[:, np.r_[33:41]]
    # Y = join_data[['SV_label']]

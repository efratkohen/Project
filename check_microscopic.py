from typing import Union, Tuple
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project import *


fname = "microscopic_data.csv"
data_fname = check_file(fname)
data = read_data(data_fname)
convert_dates_to_date_object(data)
micro_df_list = split_microscopic_to_reactor(data)
# data_m_reactor1, data_m_reactor2, data_m_reactor3, data_m_reactor4 = split_microscopic_to_reactor(data)
# data_m_reactor1.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_data_m_reactor1.xlsx", index=False, header=True)
# data.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_dataframe_filter.xlsx", index=False, header=True)
# reactor1_col = [data.columns[1:54]]
# print(reactor1_col)

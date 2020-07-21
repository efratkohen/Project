# from typing import Union, Tuple
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project import *

fname = "SVI.csv"
data_fname = check_file(fname)
data_SVI = read_data(data_fname)
data_SVI = check_SVI_values(data_SVI)
data_SVI = data_SVI.interpolate()
data_SVI_clean = SVI_calculate(data_SVI)
data_reactor1, data_reactor2, data_reactor3, data_reactor4 = split_to_reactor(data_SVI_clean)
df_list=[data_reactor1, data_reactor2, data_reactor3, data_reactor4]
SVI_label=[190.0, 160.0] 
"define between bad, reasonable and good result for SVI"
SV_label=[2.5, 3.0] 
"define between bad, reasonable and good result for SV"
for df in df_list:
    df= label_data(df, SVI_label, SV_label)
# print(data_reactor1.head)
data_reactor1.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_data_reactor1_label.xlsx", index=False, header=True)
# data_reactor2.to_excel(r"E:\python\project\export_data_reactor2_label.xlsx", index=False, header=True)
# data_reactor3.to_excel(r"E:\python\project\export_data_reactor3_label.xlsx", index=False, header=True)
# data_reactor4.to_excel(r"E:\python\project\export_data_reactor4_label.xlsx", index=False, header=True)
# fname = "microscopic_observation.csv"
# data_fname = check_file(fname)
# data_micro = read_data(data_fname)

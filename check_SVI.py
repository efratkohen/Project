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
data_reactor1, data_reactor2, data_reactor3, data_reactor4 = split_SVI_to_reactor(data_SVI_clean)
df_list=[data_reactor1, data_reactor2, data_reactor3, data_reactor4]
SVI_label=[190.0, 160.0] 
"define between bad, reasonable and good result for SVI"
SV_label=[2.5, 3.0] 
"define between bad, reasonable and good result for SV"
for df in df_list:
    df= label_data(df, SVI_label, SV_label)
fname1 = "microscopic_data.csv"
data_fname1 = check_file(fname1)
data = read_data(data_fname1)
data_m_reactor1, data_m_reactor2, data_m_reactor3, data_m_reactor4 = split_microscopic_to_reactor(data)
data_reactor1.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_data_reactor1_label.xlsx", index=False, header=True)
data_m_reactor1.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_data_m_reactor1.xlsx", index=False, header=True)
# data_reactor2.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_data_reactor2_label.xlsx", index=False, header=True)
# data_reactor3.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_data_reactor3_label.xlsx", index=False, header=True)
# data_reactor4.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_data_reactor4_label.xlsx", index=False, header=True)


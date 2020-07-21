from typing import Union, Tuple
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project import *


fname = "microscopic_observation.csv"
data_fname = check_file(fname)
data = read_data(data_fname)
# data.to_excel(r"E:\python\project\export_dataframe_filter.xlsx", index=False, header=True)
reactor1_col = [data.columns[1:54]]
print(reactor1_col)

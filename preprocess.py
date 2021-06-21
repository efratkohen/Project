import json
from os import walk

import matplotlib.pyplot as pyplot
import pandas as pd
import seaborn as sns

import column_labeler as clabel


# In[5]:

def is_holidays(vals):
    res = []
    with open("jewish_holidays.json", "r") as json_file:
        jewish_holidays = json.load(json_file)
    print(jewish_holidays)
    for val in vals:
        datetime = pd.to_datetime(val)
        val = str(datetime.year) + "_" + str(datetime.month).zfill(2) + "_" + str(datetime.day).zfill(2)
        if val in jewish_holidays:
            res.append(1)
        else:
            res.append(0)
    return res


def char_remover(char):
    if ord(char) < 33 or ord(char) > 126:
        return True
    if char == "\"":
        return True
    return False


def my_float(val):
    if isinstance(val, float):
        return val
    if isinstance(val, int):
        return float(val)
    if val == '-':
        return 0
    return float(val.replace(',', ''))


def graph_viewing(df, Y_Ammonia_cols, Y_Nit_cols, Y_turb_cols):
    corr = df.corr()
    pyplot.figure(figsize=(100, 1))
    # plot the heatmap
    sns.heatmap([corr['AmmoniaConcentration-RE1mg/L']], annot=True, xticklabels=corr.columns)

    df[Y_Ammonia_cols].replace(-1.0, float("Nan")).plot(subplots=True, legend=True, figsize=(13, 10))
    pyplot.suptitle("Ammonia", fontsize=14)
    pyplot.show()

    df[Y_Nit_cols].replace(-1.0, float("Nan")).plot(subplots=True, legend=True, figsize=(13, 10))
    pyplot.suptitle("Nitrate", fontsize=14)
    pyplot.show()

    df[Y_turb_cols].replace(-1.0, float("Nan")).plot(subplots=True, legend=True, figsize=(13, 10))
    pyplot.suptitle("Turbidity", fontsize=14)
    pyplot.show()


# In[1]

def day_first(df_day_first):
    min_index = min(df_day_first.index)
    max_index = max(df_day_first.index)
    res = (max_index - min_index).days
    if res < 15:
        return True
    return False


def add_one_hot_time(df):
    df = df.reset_index(drop=True)
    weekdays_df = pd.get_dummies(df.index.dayofweek, prefix='dayofweek')
    hours_df = pd.get_dummies(df.index.hour, prefix='hour')
    months_df = pd.get_dummies(df.index.month, prefix='month')
    # row_size = df.shape[0]
    df = pd.concat([df, weekdays_df, months_df, hours_df], axis=1)
    return df


def preprocess_df(selected_value, graphs=False):
    dfs = []

    # In[]
    f = []
    print("[LOG] Started preprocessing")
    mypath = "PartedData"
    dfs = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
    for filename in f:
        cur = mypath + "\\" + filename
        df_day_first = pd.read_csv(cur, index_col='Time', parse_dates=True, dayfirst=True)
        if day_first(df_day_first):
            dfs.append(df_day_first)
        else:
            print("DayLater", filename)
            pd.read_csv(cur, index_col='Time', parse_dates=True, dayfirst=False)

    df = pd.concat(dfs)
    df = df.sort_index()

    df = df.dropna(thresh=10, axis=0)

    df.columns = list(map(lambda x: ''.join(["" if char_remover(i) else i for i in x]), df.columns))
    for i in range(1, len(df.columns)):
        df.iloc[:, i] = df.iloc[:, i].apply(my_float)
        df.iloc[:, i] = df.iloc[:, i].astype(float)

    print("[LOG] Removing Data")
    max_number_of_nas = 2*df.shape[0] // 3
    df = df.loc[:, (df.isnull().sum(axis=0) <= max_number_of_nas)]
    interpolate = False
    if interpolate:
        df = df.interpolate(method='linear', limit_direction='forward', axis=0)
        df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
    else:
        df = df.apply(lambda x: x.fillna(-1), axis=0)
    # Extracting columns that can be labeling columns
    labeler = clabel.label_columns(df.columns)
    Xcols = [col for col in df if labeler[col][0] == clabel.INPUT]
    Y_Ammonia_cols = [col for col in df if labeler[col][0] == clabel.OUTPUT and labeler[col][1] == clabel.AMMONIA][1:]
    Y_Nit_cols = [col for col in df if labeler[col][0] == clabel.OUTPUT and labeler[col][1] == clabel.NITRATE_CONC]
    # Turbidity Usually the same always and has high peaks sometime.
    Y_turb_cols = [col for col in df if labeler[col][0] == clabel.OUTPUT and labeler[col][1] == clabel.TURBIDITY]
    # Also have peaks, more often than turbidity.
    if graphs:
        graph_viewing(df, Y_Ammonia_cols, Y_Nit_cols, Y_turb_cols)
    # todo clean weird points
    if selected_value == clabel.AMMONIA:
        col_name = 'Ammonia-StageBmg/L'
    elif selected_value == clabel.NITRATE_CONC:
        df.insert(0, col_name, df.pop(col_name))
        df = df[[Y_Nit_cols[0]] + Xcols + Y_Ammonia_cols + Y_Nit_cols[1:] + Y_turb_cols]
    elif selected_value == clabel.TURBIDITY:
        col_name = 'Turbidity-StageBNTU'
        df.insert(0, col_name, df.pop(col_name))
    else:
        raise Exception()
    return df


def data_adding():
    perc_df = pd.read_csv('PublicData/beit_dagan.csv', parse_dates=[['date', 'hour']], dayfirst=True, low_memory=False)
    perc_df.index = pd.to_datetime(perc_df.date_hour)
    perc_df = perc_df.resample('1min').asfreq()
    perc_df['date_hour'] = perc_df.index
    for col in perc_df.columns:
        perc_df[col] = perc_df[col].ffill().bfill()
    perc_df = perc_df.apply(pd.to_numeric, errors='ignore')
    perc_df.drop_duplicates(inplace=True)

    # Todo: Add Holiday lists
    # df['holiday']=is_holidays(df.index.values)
    return perc_df

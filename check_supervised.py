import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import project as pr
import time_hyper_params as thp
import supervised as sup
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

if __name__=='__main__':
    svi_df_list = thp.SVI_read_and_process()
    microscopic_df_list = thp.microscopic_data_read_and_process()
    microscopic_df_list = sup.drop_nan_rows(microscopic_df_list)
    join_data_1, join_data_2, join_data_3, join_data_4 = sup.merge_data_df_list(19, 27, microscopic_df_list, svi_df_list, 3)
    join_list = [join_data_1, join_data_2, join_data_3, join_data_4]
    X_train = pd.concat(join_list[0:3]).iloc[:,2:9]
    X_test = join_list[3].iloc[:,2:9]
    y_train = pd.concat(join_list[0:3]).iloc[:,11]
    y_test = join_list[3].iloc[:,11]
    scores_data = sup.check_K_values(20, X_train, y_train, X_test, y_test)
    sns.set()
    sns.stripplot(data=scores_data, x='k', y='score', size=10)
    plt.show()
    
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(X_train, y_train)
    # print(knn.score(X_test, y_test))


    
    # X_train.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_X_train.xlsx", index=False, header=True)
    # join_df = merge_data(19, 27, microscopic1, svi1, 3)
    # svi1.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_svi1.xlsx", index=False, header=True)
    # microscopic1.to_excel(r"E:\python\Microorganism_Effects_Analysis\export_microscopic1.xlsx", index=False, header=True)
   
  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ml_prepare import ML_prepare
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def check_K_values(k_max: int, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    k_range = range(1,k_max)
    scores_k = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores_k.append(knn.score(X_test, y_test))
    scores_data = pd.DataFrame({"k": k_range, "score": scores_k})
    return scores_data


def score_by_label (y_test, y_predict):
    """return score of prediction by label type"""
    df = pd.DataFrame(list(zip(list(y_test), y_predict)), 
               columns =['y_test', 'y_predict'])
    label_lst = ["bad", "reasonable", "good"]
    score_lst = ["score_bad", "score_reasonable", "score_good"]
    for i in range (3):
        score_lst[i] = accuracy_score(list(df.loc[df['y_test']==label_lst[i]].loc[:,'y_test']), 
                    list(df.loc[df['y_test']==label_lst[i]].loc[:,'y_predict']))
    return score_lst


# def check_delay_values(k_max: int, X_train, y_train, X_test, y_test) -> pd.DataFrame:
#     k_range = range(1,k_max)
#     scores_k = []
#     for k in k_range:
#         knn = KNeighborsClassifier(n_neighbors=k)
#         knn.fit(X_train, y_train)
#         scores_k.append(knn.score(X_test, y_test))
#     scores_data = pd.DataFrame({"k": k_range, "score": scores_k})
#     return scores_data

if __name__ == "__main__":
    delay_lst = [*range(0, 18, 3)]
    sections = ['all','total_counts', 'filaments', 'various']
    labels = ['SV_label', 'SVI_label']
    score_delay = [] 
    for label in labels:
        for section in sections:
            for delay in delay_lst:
                data = ML_prepare(delay)
                table_xy = data.get_partial_table(x_section=section,y_labels=True)
                X = table_xy.loc[:,'x']
                y = table_xy['y', label]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                knn = KNeighborsClassifier(n_neighbors=10)
                knn.fit(X_train, y_train)
                y_predict=(knn.predict(X_test))
                score = score_by_label(y_test, y_predict)
                score_delay.append(score)
                print(f"result for delay= {delay}, section= {section}, label={label} : score_bad={score[0]}, score_reasonable={score[1]}, score_good={score[2]}")


    
    # score_bad, score_reasonable, score_good = score_by_label(y_test, y_predict)
    # print(f"score_bad={score_bad}, score_reasonable={score_reasonable}, score_good={score_good}")
    
    # scores_data = check_K_values(20, x_train, y_train, x_test, y_test)
    # sns.set()
    # sns.stripplot(data=scores_data, x='k', y='score', size=10)
    # plt.ylim(0, 1)
    # plt.show()

    # print(x[0]['1':'3'].iloc[:, 0:18])
    # x[0].to_excel(r"E:\python\Microorganism_Effects_Analysis\x_0.xlsx", index=True, header=True)
    # y[0].to_excel(r"E:\python\Microorganism_Effects_Analysis\y_0.xlsx", index=True, header=True)
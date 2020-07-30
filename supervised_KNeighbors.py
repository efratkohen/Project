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


def choose_k_value(section: str, label: str, delay: int):
    """plot graph for KNeighborsClassifier and the user choose the k value by the result"""
    data = ML_prepare(delay)
    table_xy = data.get_partial_table(x_section=section,y_labels=True)
    X = table_xy.loc[:,'x']
    y = table_xy['y', label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scores_data = check_K_values(20, X_train, y_train, X_test, y_test)
    sns.set()
    sns.stripplot(data=scores_data, x='k', y='score', size=10)
    plt.ylim(0, 1)
    print("please look at the graph and choose k value. press any key to continue")
    input()
    plt.show()
    print("please insert k value")
    k = int(input())
    return k


def create_score_list(labels: list, sections: list, delay_lst: list, k: int) -> list:
    score_delay = [] 
    for label in labels:
        for section in sections:
            for delay in delay_lst:
                data = ML_prepare(delay)
                table_xy = data.get_partial_table(x_section=section,y_labels=True)
                X = table_xy.loc[:,'x']
                y = table_xy['y', label]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                knn = KNeighborsClassifier(n_neighbors = k)
                knn.fit(X_train, y_train)
                y_predict=(knn.predict(X_test))
                score_label = score_by_label(y_test, y_predict)
                score = knn.score(X_test, y_test)
                score_label.append(score)
                score_delay.append(score_label)
                # print(f"result for delay= {delay}, section= {section}, label={label} : score_bad={score_label[0]}, score_reasonable={score_label[1]}, score_good={score_label[2]}, score={score_label[3]}")
    return score_delay

def list_to_df(score_lst: list, delay_lst: list, sections: list, labels: list, score_lst_name: list ) -> pd.DataFrame:
    arr = np.array(score_lst) 
    arr1 = np.zeros([6, 32], dtype=float)
    col = 0
    for k in range(8):
        for i in range(6):
            for z in range(4):
                arr1[i, z+k*4] = arr[col, z]
            col = col + 1  
    index = pd.MultiIndex.from_product([delay_lst],
                                    names=['delay'])
    columns = pd.MultiIndex.from_product([labels, sections, score_lst_name],
                                    names=['label', 'section', 'score_type'])
    df = pd.DataFrame(arr1, index=index, columns=columns)
    return df 


# def save_plot(score_df: pd.DataFrame, delay_lst: list, sections: list, labels: list, score_lst_name: list, delay_lst: list):
#     sns.set()
#     for label in labels:
#         for section in sections:
#             for score_type in score_lst_name:
#                 sns.stripplot(data=score_df[label, section, score_type], x='delay', y='{label}, {section}, {score_type}', size=10)
#                 score_df.plot(kind='scatter',x='delay',y=[label, section, score_type])
#                 plt.ylim(0, 1)
                # plt.savefig(f"KNeighbors_plot/{label}_{section}_{score_type}.png")


if __name__ == "__main__":
    delay_lst = [*range(0, 18, 3)]
    sections = ['all','total_counts', 'filaments', 'various']
    labels = ['SV_label', 'SVI_label']
    score_lst_name = ['bad_s', 'reasonable_s', 'good_s', 'score']
    # k = choose_k_value('filaments', 'SVI_label', 12)
    k = 9 # erase 
    score_lst = create_score_list(labels, sections, delay_lst, k)
    score_df = list_to_df(score_lst, delay_lst, sections, labels, score_lst_name)
    print(score_df)
    

    # x[0].to_excel(r"E:\python\Microorganism_Effects_Analysis\x_0.xlsx", index=True, header=True)
    # y[0].to_excel(r"E:\python\Microorganism_Effects_Analysis\y_0.xlsx", index=True, header=True)
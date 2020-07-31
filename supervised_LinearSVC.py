import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ml_prepare import ML_prepare
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def score_by_label (y_test, y_predict):
    """
    return score of prediction by label type

    Parameters
    ----------
    y_test : pd.DataFrame
        y value to predict
    y_predict : list
        y value predicted
    

    Attributes
    ----------
    score_list : list
        score list of score predictions for labels bad, reasonable and good
    """
    
    df = pd.DataFrame(list(zip(list(y_test), y_predict)), 
               columns =['y_test', 'y_predict'])
    label_lst = ["bad", "reasonable", "good"]
    score_lst = ["score_bad", "score_reasonable", "score_good"]
    for i in range (3): 
        score_lst[i] = accuracy_score(list(df.loc[df['y_test']==label_lst[i]].loc[:,'y_test']), 
                    list(df.loc[df['y_test']==label_lst[i]].loc[:,'y_predict']))
    return score_lst


def create_score_list(labels: list, sections: list, delay_lst: list) -> list:
    score_delay = [] 
    for label in labels:
        for section in sections:
            for delay in delay_lst:
                data = ML_prepare(delay)
                table_xy = data.get_partial_table(x_section=section,y_labels=True)
                X = table_xy.loc[:,'x']
                y = table_xy['y', label]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                clf = make_pipeline(StandardScaler(),
                     LinearSVC(random_state=0, tol=1e-5, max_iter=100000))
                clf.fit(X_train, y_train)
                y_predict=(clf.predict(X_test))
                score_label = score_by_label(y_test, y_predict)
                score = clf.score(X_test, y_test)
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
    score_lst = create_score_list(labels, sections, delay_lst)
    score_df = list_to_df(score_lst, delay_lst, sections, labels, score_lst_name)
    #print(score_df)
    

    score_df.to_excel(r"E:\python\Microorganism_Effects_Analysis\SVC_rendom_state_0_max_iter_100000.xlsx", index=True, header=True)
    # y[0].to_excel(r"E:\python\Microorganism_Effects_Analysis\y_0.xlsx", index=True, header=True)
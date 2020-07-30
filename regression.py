from ml_prepare import ML_prepare
from collections import namedtuple
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pca_plot(table_xy: pd.DataFrame, section: str, ax_i, color_col="SVI"):
    x_only = table_xy.loc[:, "x"]

    pca_model = make_pipeline(StandardScaler(), PCA(n_components=2))
    pca_model.fit(x_only)

    X_2D = pca_model.transform(x_only)
    pca_dict = dict(PCA1=X_2D[:, 0], PCA2=X_2D[:, 1])
    pca_results = pd.DataFrame(pca_dict)

    color_series = table_xy.loc[:, ("y", color_col)].reset_index(drop=True)

    pca_results["color"] = color_series

    g = sns.scatterplot(data=pca_results, x="PCA1", y="PCA2", hue="color", ax=ax_i)
    g.legend_.remove()
    g.set(title=f"{section} colored by {color_col}")


def create_section_and_PCA(data: ML_prepare, labled: bool = False):
    section_lst = ["all", "filaments", "total_counts", "various"]
    fig, ax = plt.subplots(4, 2)
    for i in range(len(section_lst)):
        table_xy = data.get_partial_table(x_section=section_lst[i], y_labels=labled)
        y_cols = table_xy.loc[:, "y"].columns.tolist()
        for j in range(2):
            ### model on y = y_cols[j]
            pca_plot(
                table_xy, color_col=y_cols[j], section=section_lst[i], ax_i=ax[i, j]
            )
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.suptitle(
        f"PCA of groups, colored by output, delay = {data.delay} days",
        fontsize=20,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

def insert_scores_to_namedtuple(scores_lst:list):
    Tup_scores = namedtuple(
        "Tup_scores",
        [
            "all_sv",
            "all_svi",
            "filaments_sv",
            "filaments_svi",
            "total_counts_sv",
            "total_counts_svi",
            "various_sv",
            "various_svi",
        ]
    )
    tup_scores = Tup_scores(*scores_lst)
    return tup_scores


def loop_over_sections_and_y(func, data: ML_prepare):
    scores_lst = []
    section_lst = ["all", "filaments", "total_counts", "various"]
    for i in range(len(section_lst)):
        table_xy = data.get_partial_table(x_section=section_lst[i], y_labels=False)
        y_cols = table_xy.loc[:, "y"].columns.tolist()
        for j in range(2):
            score = func(X=table_xy.loc[:, "x"], y=table_xy.loc[:, ("y", y_cols[j])])
            scores_lst.append(score)
            # print(
            #     f"Model: {func.__name__}\tfor section {section_lst[i]} y = {y_cols[j]}\t\tscore = {score:.3f}"
            # )
    tup_scores = insert_scores_to_namedtuple(scores_lst)
    return tup_scores


def Lasso_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
    )
    lasso_model = linear_model.Lasso(alpha=1)
    lasso_model.fit(X_train, y_train)
    score = lasso_model.score(X_test, y_test)
    return score

def SVR_linear(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
    )
    svr_l = make_pipeline(StandardScaler(), SVR())
    svr_l.fit(X_train, y_train)
    score = svr_l.score(X_test, y_test)
    return score


def get_scores_of_model(model_func=Lasso_model, print_flag=True):
    
    print(f'model: {model_func.__name__}')
    scores_by_delay_dict = {}
    for delay in range(1,13):
        data = ML_prepare(delay=delay)
        scores_by_delay_dict[delay] = loop_over_sections_and_y(func=model_func, data=data)

    if print_flag:
        for delay in scores_by_delay_dict:
            tup_delay = scores_by_delay_dict[delay]
            max_score = max(tup_delay)
            name = [tup_delay._fields[i] for i in range(len(tup_delay)) if tup_delay[i]==max_score]
            print(f'max score for delay {delay}\t {max_score:.2f}, for {name[0]}')
    
    return scores_by_delay_dict
    

if __name__ == "__main__":
    data = ML_prepare(delay=4)
    create_section_and_PCA(data, labled=True)
    create_section_and_PCA(data, labled=False)

    # scores_by_delay_dict_Lasso = get_scores_of_model(model_func=Lasso_model)
    # scores_by_delay_dict_Lasso = get_scores_of_model(model_func=SVR_linear)



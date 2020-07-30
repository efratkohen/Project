from ml_prepare import ML_prepare
from collections import namedtuple
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import linear_model
from sklearn.linear_model import ElasticNet, Ridge
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

def regr_model_func(X, y, reg_model):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
    )
    reg_model.fit(X_train, y_train)
    score = reg_model.score(X_test, y_test)
    return score

def loop_over_sections_and_y(data: ML_prepare, regr_model):
    scores_lst = []
    section_lst = ["all", "filaments", "total_counts", "various"]
    for i in range(len(section_lst)):
        table_xy = data.get_partial_table(x_section=section_lst[i], y_labels=False)
        y_cols = table_xy.loc[:, "y"].columns.tolist()
        for j in range(2):
            X = table_xy.loc[:, "x"]
            y = table_xy.loc[:, ("y", y_cols[j])]
            score = regr_model_func(X, y, regr_model)
            scores_lst.append(score)

    tup_scores = insert_scores_to_namedtuple(scores_lst)
    return tup_scores

def get_scores_of_model(regr_model, model_name:str, print_flag:bool=True):
    
    print(f'\nmodel: {model_name}')
    scores_by_delay_dict = {}
    for delay in range(1,13):
        data = ML_prepare(delay=delay)
        scores_by_delay_dict[delay] = loop_over_sections_and_y(data=data, regr_model=regr_model)

    if print_flag:
        for delay in scores_by_delay_dict:
            tup_delay = scores_by_delay_dict[delay]
            max_score = max(tup_delay)
            name = [tup_delay._fields[i] for i in range(len(tup_delay)) if tup_delay[i]==max_score]
            print(f'max score for delay {delay}\t {max_score:.2f}, for {name[0]}')
    
    return scores_by_delay_dict

def get_scores_of_all_models(models_dict: dict, print_flag:bool=True):
    '''
    '''
    scores_models_dict = {}
    for model in models_dict:
        model_name = models_dict[model]
        res = get_scores_of_model(model, model_name)
        scores_models_dict[model_name] = res
    return scores_models_dict


    
def run_models_on_same_data_and_plot(models_dict, X, y, x_name:str):
    print(f'Predicting {y.name[1]} by {x_name}:')
    for model in models_dict:
        model_name = models_dict[model]
        run_model_and_plot(model, model_name, X, y, x_name)
    
def run_model_and_plot(regr_model, model_name:str, X, y, x_name:str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
    )
    regr_model.fit(X_train, y_train)
    y_predicted = regr_model.predict(X)
    reg_plot(y, y_predicted, model_name)

def reg_plot(x_axis, y_axis, model_name):
    
    

    sns.set()
    fig1, ax1 = plt.subplots()
    sns.regplot(x_axis, y_axis, ax=ax1)
    ax1.set_xlabel('real values')
    ax1.set_ylabel('predicted')
    fig1.suptitle(f'model {model_name}', fontsize=16)
    plt.show()


if __name__ == "__main__":
    data = ML_prepare(delay=4)
    # create_section_and_PCA(data, labled=True)
    # create_section_and_PCA(data, labled=False)

    ### create models
    las = linear_model.Lasso(alpha=1)
    elast = ElasticNet(random_state=0)
    ridge = Ridge(alpha=1.0)
    svr_rbf = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
    svr_lin = make_pipeline(StandardScaler(), SVR(kernel='linear'))

    models_dict = {las:'Lasso', elast:'ElasticNet', ridge:'Ridge Regression', svr_lin:'SVR (lin)', svr_rbf:'SVR (rbf)'}

    # get scores for all models for all sections:
    # scores_models_dict = get_scores_of_all_models(models_dict, print_flag=True)

    # best looks SVI for filaments, after 3 days:
    data2 = ML_prepare(delay=3)
    filaments_table = data2.get_partial_table(x_section='filaments',y_labels=False)
    filaments_x = filaments_table.loc[:,'x']
    filaments_svi = filaments_table.loc[:,('y','SVI')]
    run_models_on_same_data_and_plot(models_dict, filaments_x, filaments_svi, 'filaments')



    


    
    # filaments = data.get_partial_table(x_section='filaments', y_labels=False)
    # filaments_x = filaments.loc[:,'x']
    # fils_melt = pd.melt(filaments_x)

    # filaments_xy = pd.concat([filaments_x, filaments.loc[:,'y']], axis=1)

    # sns.pairplot(filaments_xy)








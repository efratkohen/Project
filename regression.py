from ml_prepare import ML_prepare
from collections import namedtuple
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats


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


def insert_scores_to_namedtuple(scores_lst: list):
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
        ],
    )
    tup_scores = Tup_scores(*scores_lst)
    return tup_scores








def get_scores_of_all_models(models_dict: dict, delay_range=range(1,13), print_flag: bool = True):
    """
    """
    scores_models_dict = {}
    for model in models_dict:
        model_name = models_dict[model]
        res = get_scores_of_model(model, model_name, delay_range, print_flag)
        scores_models_dict[model_name] = res
    return scores_models_dict


def get_scores_of_model(regr_model, model_name: str, delay_range, print_flag: bool = True):

    scores_by_delay_dict = {}
    for delay in delay_range:
        data = ML_prepare(delay=delay)
        scores_by_delay_dict[delay] = loop_over_sections_and_y(
            data=data, regr_model=regr_model
        )

    if print_flag:
        print(f"\nmodel: {model_name}")
        for delay in scores_by_delay_dict:
            tup_delay = scores_by_delay_dict[delay]
            max_score = max(tup_delay)
            name = [
                tup_delay._fields[i]
                for i in range(len(tup_delay))
                if tup_delay[i] == max_score
            ]
            print(f"max score for delay {delay}\t {max_score:.2f}, for {name[0]}")

    return scores_by_delay_dict

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


def regr_model_func(X, y, reg_model):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
    )
    reg_model.fit(X_train, y_train)
    score = reg_model.score(X_test, y_test)
    return score



def run_models_on_same_data_and_plot(models_dict, X, y, x_name: str):
    fig1, ax1 = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    plt.xlim((65, 210))
    plt.ylim((65, 210))
    fig1.suptitle(
        f"Predicting {y.name[1]} by {x_name}, delay = 3 days:", fontsize=20, y=1.05
    )

    ax_count = 0
    for model in models_dict:
        model_name = models_dict[model]
        run_model_and_plot(model, model_name, X, y, x_name, ax1[ax_count])
        ax_count += 1

    fig1.text(0.5, 0.0, "Real values", ha="center", va="center", fontsize=14)
    fig1.text(0.0, 0.5, "Predicted", ha="center", va="center", fontsize=14, rotation=90)
    plt.tight_layout()
    fig1.savefig("SVI_filaments_regression.png", dpi=150, bbox_inches="tight")
    plt.show()


def run_model_and_plot(regr_model, model_name: str, X, y, x_name: str, ax_i):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
    )
    regr_model.fit(X_train, y_train)
    y_predicted = regr_model.predict(X_test)
    score = regr_model.score(X_test, y_test)

    reg_plot(y_test, y_predicted, model_name, score, ax_i)


def reg_plot(x_axis, y_axis, model_name, score, ax_i):
    """
    """
    sns.set()
    sns.regplot(x_axis, y_axis, ax=ax_i)
    r, pvalue = scipy.stats.pearsonr(x_axis, y_axis)

    Mean_SE = np.mean((x_axis - y_axis) ** 2)

    ax_i.text(
        70,
        205,
        f"Pearson R: {r:.3f}\nP-value: {pvalue:.3e}\nMean SE: {Mean_SE:.2f}",
        va="top",
        ha="left",
        fontsize=10,
    )
    ax_i.text(135, 75, f"Score (R^2): {score:.2f}", fontsize=13, ha="left")
    ax_i.set_title(f"{model_name}", fontsize=16)
    ax_i.set_xlabel("")

def create_list_of_tidy_df_by_day(scores_models_dict: dict, delay_range):
    '''
    Each subplot is a delay. In each there is all the results of a given section for all 5 models.
    '''
    days_df_dict = {}
    categories = scores_models_dict['Lasso'][3]._fields

    for day in delay_range:
        df_day = pd.DataFrame(columns=['score','section','sv_svi','model'])
        for model in scores_models_dict:
            model_tup_res = scores_models_dict[model][day]
            df_model = pd.DataFrame()
            df_model['score'] = model_tup_res
            df_model['section']=['all','all','filaments','filaments','total_counts','total_counts','various','various']
            df_model['sv_svi'] = 4*['sv','svi']
            df_model['model'] = 8*[model]
            df_day = pd.concat([df_day, df_model], ignore_index=True)

        days_df_dict[day] = df_day
    
    return days_df_dict

def days_swarmplot(days_df_dict):
    days_range = range(2,8)
    fig2, ax2 = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    plt.ylim((-0.3,0.6))
    fig2.suptitle(
        f"Score for all sections and models by day", fontsize=20, y=1.05
    )
    fig2.text(0.5, 0.0, "Section", ha="center", va="center", fontsize=20)
    fig2.text(0.0, 0.5, "Score", ha="center", va="center", fontsize=20, rotation=90)
    

    ax_count = 0
    for day_num in days_range:
        ax_row = (ax_count)//3
        ax_col = (ax_count)%3
        swarmplot_of_day(days_df_dict[day_num], day_num, ax_i=ax2[ax_row, ax_col])
        ax_count += 1
    
    
    plt.tight_layout()
    fig2.savefig("Swarm_plot_by_day.png", dpi=150, bbox_inches="tight")
    plt.show()


def swarmplot_of_day(df_day:pd.DataFrame, day_num:int, ax_i):
    sns.set()
    sns.swarmplot(x="section", y="score", hue="sv_svi",
                   data=df_day, palette="Set2", dodge=True, ax=ax_i)

    ax_i.set_title(f"delay: {day_num} days", fontsize=16)
    ax_i.set_xlabel("")
    ax_i.set_ylabel("")



def create_models_dict():
    las = linear_model.Lasso(alpha=1)
    elast = ElasticNet(random_state=0)
    ridge = Ridge(alpha=1.0)
    svr_rbf = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
    svr_lin = make_pipeline(StandardScaler(), SVR(kernel="linear"))

    models_dict = {
        las: "Lasso",
        elast: "ElasticNet",
        ridge: "Ridge Regression",
        svr_lin: "SVR (lin)",
        svr_rbf: "SVR (rbf)",
    }
    return models_dict

def run_models_on_filaments_svi_3days(models_dict:dict):
    data2 = ML_prepare(delay=3)
    filaments_table = data2.get_partial_table(x_section="filaments", y_labels=False)
    filaments_x = filaments_table.loc[:, "x"]
    filaments_svi = filaments_table.loc[:, ("y", "SVI")]

    run_models_on_same_data_and_plot(
        models_dict, filaments_x, filaments_svi, "filaments"
    )


if __name__ == "__main__":
    data = ML_prepare(delay=4)
    #### PCA
    create_section_and_PCA(data, labled=True)
    create_section_and_PCA(data, labled=False)

    #### Run all models on all delays, all sections 
    models_dict = create_models_dict()
    delay_range = range(1,13)

    # get scores for all models for all sections:
    scores_models_dict = get_scores_of_all_models(models_dict, delay_range= delay_range, print_flag=False)
    # turn to list of long dfs
    days_df_dict = create_list_of_tidy_df_by_day(scores_models_dict, delay_range)

    # plot them by days
    days_swarmplot(days_df_dict)

    ###### best looks SVI for filaments, after 3 days:
    run_models_on_filaments_svi_3days(models_dict)
    



    







    ###### for me later
    # X_train, X_test, y_train, y_test = train_test_split(
    #     filaments_x, filaments_svi, test_size=0.25, random_state=42,
    # )
    # ridge.fit(X_train, y_train)
    # y_predicted = ridge.predict(filaments_x)

    # mse = np.mean((ridge.predict(X_test) - y_test)**2)


    # delay_range = range(2,8)
    # days_df_dict = {}
    # categories = scores_models_dict['Lasso'][3]._fields

    # day = 3
    # df_day = pd.DataFrame(columns=['score','section','sv_svi','model']) # remove
    # # df_day = pd.DataFrame(index=list(categories))
    # for model in scores_models_dict:
    #     model_tup_res = scores_models_dict[model][day]
    #     df_model = pd.DataFrame()
    #     df_model['score'] = model_tup_res
    #     df_model['section']=['all','all','filaments','filaments','total_counts','total_counts','various','various']
    #     df_model['sv_svi'] = 4*['sv','svi']
    #     df_model['model'] = 8*[model]
    #     df_day = pd.concat([df_day, df_model], ignore_index=True)


    
    # return days_df_dict

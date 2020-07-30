from ml_prepare import ML_prepare
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def pca_plot(table_xy: pd.DataFrame, section: str, ax_i, color_col="SVI"):
    x_only = table_xy.loc[:, "x"]

    pca_model = PCA(n_components=2)
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


def loop_over_sections_and_y(func):
    scores_lst = []
    section_lst = ["all", "filaments", "total_counts", "various"]
    for i in range(len(section_lst)):
        table_xy = data.get_partial_table(x_section=section_lst[i], y_labels=False)
        y_cols = table_xy.loc[:, "y"].columns.tolist()
        for j in range(2):
            score = func(X=table_xy.loc[:, "x"], y=table_xy.loc[:, ("y", y_cols[j])])
            scores_lst.append(score)
            print(
                f"Model: {func.__name__}\tfor section {section_lst[i]} y = {y_cols[j]}\t\tscore = {score:.3f}"
            )
    return scores_lst


def Lasso_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
    )
    lasso_model = linear_model.Lasso(alpha=1)
    lasso_model.fit(X_train, y_train)
    score = lasso_model.score(X_test, y_test)
    return score


if __name__ == "__main__":
    data = ML_prepare(delay=3)
    # create_section_and_PCA(data, labled=True)
    # create_section_and_PCA(data, labled=False)

    scores_3 = loop_over_sections_and_y(func=Lasso_model)


    scores_dict = {}
    # for delay in range(2,9):
    #     data = ML_prepare(delay=delay)
    #     #

    #     scores_dict[delay] = Lasso_model_scores(data)

    # t_all = data.get_partial_table(x_section="all")
    # t_filaments = data.get_partial_table(x_section="filaments")
    # t_total = data.get_partial_table(x_section="total_counts")
    # t_various = data.get_partial_table(x_section="various")

    #####
    # X_train, X_test, y_train, y_test = train_test_split(
    #     t_filaments.loc[:, "x"],
    #     t_filaments.loc[:, ("y", "SVI")],
    #     test_size=0.25,
    #     random_state=42,
    # )

    # lasso_model = linear_model.Lasso(alpha=1)
    # lasso_model.fit(X_train, y_train)
    # score = lasso_model.score(X_test, y_test)
    # score_train = lasso_model.score(X_train, y_train)
    # print(f"score = {score}")

    # days = 4
    # for section filaments y = SVI, score = 0.35173946953475366

    # days = 3
    # for section filaments y = SVI, score = 0.454


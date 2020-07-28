import ml_prepare as ml_p
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier


def check_K_values(k_max: int, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    k_range = range(1,k_max)
    scores_k = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores_k.append(knn.score(X_test, y_test))
    scores_data = pd.DataFrame({"k": k_range, "score": scores_k})
    return scores_data


if __name__ == "__main__":
    data = ml_p.ML_prepare()
    delay = [0, 3, 6, 9, 12, 15]
    x = ["x_0", "x_3", "x_6", "x_9", "x_12", "x_15"]
    y = ["y_0", "y_3", "y_6", "y_9", "y_12", "y_15"]
    for i in range(6):
        x[i], y[i] = data.create_x_y_delayed(days_delay=delay[i])
    x_train = x[0].loc['1':'3'].iloc[:, 18:26]
    x_test = x[0].loc['4'].iloc[:, 18:26]
    y_train = y[0].loc['1':'3', "SV_label"]
    y_test = y[0].loc['4', "SV_label"]
    print(x_train, y_train)
    scores_data = check_K_values(20, x_train, y_train, x_test, y_test)
    sns.set()
    sns.stripplot(data=scores_data, x='k', y='score', size=10)
    sns.plt.ylim(0, 1)
    plt.show()

    # print(x[0]['1':'3'].iloc[:, 0:18])
    # x[0].to_excel(r"E:\python\Microorganism_Effects_Analysis\x_0.xlsx", index=True, header=True)
    # y[0].to_excel(r"E:\python\Microorganism_Effects_Analysis\y_0.xlsx", index=True, header=True)
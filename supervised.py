import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ml_prepare import ML_prepare
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


def check_K_values(k_max: int, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    plot figures of score results
    x axis - delay
    y axis - score

    Parameters
    ----------
    k_max: int
        max value to check
    X_train: pd.DataFrame
        X data to train the model
    y_train: pd.Series
        y labels to train the model
    X_test: pd.DataFrame
        X data to test the model
    y_test: pd.Series
        y label to test the model
    
    return
    ----------
    df : pd.DataFrame
        DataFrame with k_range and scores_k
    """
    if k_max <= 0:
        raise ValueError("Please supply k_max value > 0 ")
    k_range = range(1,k_max)
    scores_k = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores_k.append(knn.score(X_test, y_test))
    scores_data = pd.DataFrame({"k": k_range, "score": scores_k})
    return scores_data


def score_by_label (y_test, y_predict):
    """
    calculate score of prediction by label type: bas, reasonable and good

    Parameters
    ----------
    y_test : pd.Series
        y value to predict
    y_predict : list
        y value predicted
    

    return
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


def choose_k_value(section: str, label: str, delay: int):
    """
    plot graph of scores_k by k_range for KNeighbors model.
    The user need to choose the k value by the result.

    Parameters
    ----------
    section : str
        microorganisms section - should be: all or total_counts or filaments or various
    label : str
        label - should be: SV_label or SVI_label
    delay : int
        delay between the microscopic test which took place and the SVI test

    return
    ----------
    k : int
        k choosen by the user
    """
    if section not in {'all', 'total_counts', 'filaments', 'various'}:
        raise ValueError("Please supply a valid section value: 'all','total_counts', 'filaments', 'various' ")
    if label not in {'SV_label', 'SVI_label'}:
        raise ValueError("Please supply a valid label value: 'SV_label', 'SVI_label' ")
    data = ML_prepare(delay)
    table_xy = data.get_partial_table(x_section=section,y_labels=True)
    X = table_xy.loc[:,'x']
    y = table_xy['y', label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scores_data = check_K_values(20, X_train, y_train, X_test, y_test)
    sns.set()
    sns.stripplot(data=scores_data, x='k', y='score', size=10)
    plt.ylim(0, 1)
    print("please look at the graph and choose k value. press enter to continue")
    input()
    plt.show()
    print("please insert k value")
    k = int(input())
    return k


def create_score_list_Knn(labels: list, sections: list, delay_lst: list, k: int) -> list:
    """
    create KNeighbors score list of all the prediction by label type, section of microorganism and delay

    Parameters
    ----------
    labels : list
        list of labels: SV and SVI
    sections : list
        list of microorganisms sections: all, total_counts, filaments and various
    delay_lst : list
        list of delays between the microscopic test which took place and the SVI test
    K : int
        k value for KNeighbors model

    return
    ----------
    score_delay : list
        score list of score predictions for each combination of label, section and delay
    """
    
    if k <= 0:
        raise ValueError("Please supply k value > 0 ")
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
    return score_delay

def create_score_list_SVC(labels: list, sections: list, delay_lst: list) -> list:
    """
    create LinearSVC score list of all the prediction by label type, section of microorganism and delay

    Parameters
    ----------
    labels : list
        list of labels: SV and SVI
    sections : list
        list of microorganisms section: all, total_counts, filaments and various
    delay_lst : list
        list of delays between the microscopic test which took place and the SVI test

    return
    ----------
    score_delay : list
        score list of score predictions for each combination of label, section and delay
    """
    
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
    """
    create multi index pd.DataFrame of all score results in shape of (x,y)
    x = rows of delay values
    y = columns of multiply 2 label types, number of sections and 4 score results type

    Parameters
    ----------
    score_lst : list
        list of all scores
    delay_lst : list
        list of delays between the microscopic test which took place and the SVI test
    sections : list
        list of microorganisms section: all, total_counts, filaments and various
    labels : list
        list of labels: SV and SVI
    score_lst_name : list
        list of score types: bad_s, reasonable_s, good_s and model score

    return
    ----------
    df : pd.DataFrame
        multi index pd.DataFrame of all score results in shape of (6,36)
    """
    
    arr = np.array(score_lst) 
    arr_outlet = np.zeros([len(delay_lst), len(sections)*len(labels)*len(score_lst_name)], dtype=float)
    col = 0
    for k in range(len(sections)*len(labels)):
        for i in range(len(delay_lst)):
            for z in range(len(score_lst_name)):
                arr_outlet[i, z+k*4] = arr[col, z]
            col = col + 1  
    index = pd.MultiIndex.from_product([delay_lst],
                                    names=['delay'])
    columns = pd.MultiIndex.from_product([labels, sections, score_lst_name],
                                    names=['label', 'section', 'score_type'])
    df = pd.DataFrame(arr_outlet, index=index, columns=columns)
    return df 


def save_plot_Knn(score_df: pd.DataFrame, delay_lst: list, sections: list, labels: list, score_lst_name: list):
    """
    plot figures of score results
    x axis - delay
    y axis - score

    Parameters
    ----------
    score_df : pd.DataFrame
        multi index pd.DataFrame of all score results
    delay_lst : list
        list of delays between the microscopic test which took place and the SVI test
    sections : list
        list of microorganisms section: all, total_counts, filaments and various
    labels : list
        list of labels: SV and SVI
    score_lst_name : list
        list of score types: bad_s, reasonable_s, good_s and model score
    """
    
    for label in labels:
        fig, ax = plt.subplots(1 , len(sections), figsize=(14,4), sharey=True)
        fig.suptitle(f'scores of KNeighbors prediction for {label}', fontsize=20)
        fig.text(0.5, 0.0, "Delay (days)", ha="center", va="center", fontsize=14)
        fig.text(0.0, 0.5, "Score", ha="center", va="center", fontsize=14, rotation=90)
        plt.ylim(0, 1)
        section_count = 0
        for section in sections:
            data = score_df.loc[:,(label,section)]
            colors = ['r', 'g', 'b', 'y']
            for i in range(len(sections)):
                ax[section_count].scatter(data.index.levels[0], y=data.loc[:,score_lst_name[i]], color=colors[i], label=score_lst_name[i])
                ax[section_count].set_title(f'{section}')
                ax[section_count].set_xticks(delay_lst)
            section_count+= 1
        plt.legend(loc='upper right')
        plt.tight_layout()
        fig.savefig(f"figures/KNeighbors/KNeighbors_{label}.png", bbox_inches="tight")

def save_plot_SVC(score_df: pd.DataFrame, delay_lst: list, sections: list, labels: list, score_lst_name: list):
    """
    plot figures of score results
    x axis - delay
    y axis - score

    Parameters
    ----------
    score_df : pd.DataFrame
        multi index pd.DataFrame of all score results
    delay_lst : list
        list of delays between the microscopic test which took place and the SVI test
    sections : list
        list of microorganisms section: all, total_counts, filaments and various
    labels : list
        list of labels: SV and SVI
    score_lst_name : list
        list of score types: bad_s, reasonable_s, good_s and model score
    """
    
    for label in labels:
        fig, ax = plt.subplots(1 , len(sections), figsize=(14,4), sharey=True)
        fig.suptitle(f'scores of LinearSVC prediction for {label}', fontsize=20)
        fig.text(0.5, 0.0, "Delay (days)", ha="center", va="center", fontsize=14)
        fig.text(0.0, 0.5, "Score", ha="center", va="center", fontsize=14, rotation=90)
        plt.ylim(0, 1)
        section_count = 0
        for section in sections:
            data = score_df.loc[:,(label,section)]
            colors = ['r', 'g', 'b', 'y']
            for i in range(len(sections)):
                ax[section_count].scatter(data.index.levels[0], y=data.loc[:,score_lst_name[i]], color=colors[i], label=score_lst_name[i])
                ax[section_count].set_title(f'{section}')
                ax[section_count].set_xticks(delay_lst)
            section_count+= 1
        plt.legend(loc='upper right')
        plt.tight_layout()
        fig.savefig(f"figures/SVC/LinearSVC_{label}.png", bbox_inches="tight")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix. For classification problems
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def confusion_matrix_SVC(label: str, section: str, delay: int):
    data = ML_prepare(delay)
    table_xy = data.get_partial_table(x_section=section,y_labels=True)
    X = table_xy.loc[:,'x']
    y = table_xy['y', label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = make_pipeline(StandardScaler(),
            LinearSVC(random_state=0, tol=1e-5, max_iter=100000))
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    le = preprocessing.LabelEncoder()
    le.fit(list(y_pred))
    y_pred = le.transform(y_pred)
    y_test = le.transform(np.array(y_test))
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=le.classes_, normalize=False,
                        title='Confusion matrix, without normalization')
    plt.savefig(f"figures/SVC/LinearSVC_Confusion matrix, without normalization.png", bbox_inches="tight")


if __name__ == "__main__":
    delay_lst = [*range(0, 14, 1)]
    #sections = ['all', 'total_counts', 'filaments', 'various']
    sections = ['all', 'filaments', 'various']
    labels = ['SV_label', 'SVI_label']
    score_lst_name = ['bad_s', 'reasonable_s', 'good_s', 'score']
    model = ["Knn", "SVC"]
    #k = choose_k_value('filaments', 'SVI_label', 6)
    k = 7 # erase 
    # Run Knn model
    score_lst_Knn = []
    score_lst_Knn = create_score_list_Knn(labels, sections, delay_lst, k)
    score_df_Knn = list_to_df(score_lst_Knn, delay_lst, sections, labels, score_lst_name)
    save_plot_Knn(score_df_Knn, delay_lst, sections, labels, score_lst_name)
    
    # Run SVC model
    score_lst_SVC = []
    score_lst_SVC = create_score_list_SVC(labels, sections, delay_lst) 
    score_df_SVC = list_to_df(score_lst_SVC, delay_lst, sections, labels, score_lst_name)
    save_plot_SVC(score_df_SVC, delay_lst, sections, labels, score_lst_name)
    confusion_matrix_SVC('SVI_label', 'filaments', 7)
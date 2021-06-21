import traceback
import numpy as np
from matplotlib import pyplot, pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    median_absolute_error,
    roc_curve,
    auc,
    f1_score,
    precision_recall_curve,
    r2_score,
)
from sklearn.metrics import confusion_matrix
import column_labeler as clabel
from math import sqrt


def calc_best_f1(Ytest, Yhat, selected_value=clabel.AMMONIA):
    max_val = 0
    best_i = 0
    for i in range(1, 100):
        accuracy = f1_score(Ytest, (Yhat > 0.01 * i).astype(int))
        if accuracy > max_val:
            max_val = accuracy
            best_i = i
    f1_score(Ytest, (Yhat > 0.01 * best_i).astype(int))

    return max_val


def calc_rmse(Ytest, Yhat, graph=(20, 15)):
    rmse = sqrt(mean_squared_error(Ytest, Yhat))
    if graph:
        print("RMSE", rmse)
        pyplot.figure(figsize=graph)
        pyplot.plot(Yhat, label="predictions")
        pyplot.plot(Ytest, label="real")
        pyplot.legend()
        # import datetime
        pyplot.show()
        # pyplot.savefig("Images\\%s" % str(datetime.datetime.now()))
    return rmse


def calc_mape(Ytest, Yhat, graph=True):
    return np.mean(np.abs((Ytest - Yhat) / Ytest)) * 100


def calc_mae(Ytest, Yhat, graph=True):
    return median_absolute_error(Ytest, Yhat)


def calc_rsquared(Ytest, Yhat, graph=True):
    # R-squared
    return r2_score(Ytest, Yhat)


def calc_tp_fp_rate(Ytest, Yhat, selected_value, binary=False, graph=True):
    global y_not_bad_real, y_not_bad_hat
    if binary:
        y_not_bad_hat = Yhat.astype(int)
        y_not_bad_real = Ytest.astype(int)
    else:
        mdict = clabel.limits[selected_value]
        good_limit = mdict[clabel.GOOD]
        not_bad_limit = mdict[clabel.NOT_BAD]
        y_good_hat = Yhat > good_limit
        y_good_real = Ytest > good_limit
        y_not_bad_hat = Yhat > not_bad_limit
        y_not_bad_real = Ytest > not_bad_limit
    if graph:
        print(confusion_matrix(y_not_bad_real, y_not_bad_hat))

    res = confusion_matrix(y_not_bad_real, y_not_bad_hat).ravel()
    if len(res) > 1:
        return res
    return res[0], 0, 0, 0


def calc_best_accuracy(Ytest, Yhat, selected_value=clabel.AMMONIA):
    max_val = 0
    best_i = 0
    for i in range(1, 100):
        tn, fp, fn, tp = calc_tp_fp_rate(
            Ytest,
            (Yhat > 0.01 * i).astype(int),
            selected_value=selected_value,
            binary=True,
            graph=False,
        )
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        if accuracy > max_val:
            max_val = accuracy
            best_i = i
    calc_tp_fp_rate(
        Ytest,
        (Yhat > 0.01 * best_i).astype(int),
        selected_value=selected_value,
        binary=True,
        graph=True,
    )

    return max_val


def roc(Ytest, Yhat, graph=False):
    fpr, tpr, threshold = roc_curve(Ytest, Yhat)
    roc_auc = auc(fpr, tpr)
    # method I: plt
    if graph:
        pyplot.title("Receiver Operating Characteristic")
        pyplot.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        pyplot.legend(loc="lower right")
        pyplot.plot([0, 1], [0, 1], "r--")
        pyplot.xlim([0, 1])
        pyplot.ylim([0, 1])
        pyplot.ylabel("True Positive Rate")
        pyplot.xlabel("False Positive Rate")
        pyplot.show()
    return fpr, tpr, threshold, roc_auc


def calc_histogram(Ytest, Yhat):
    plt.figure(figsize=(15, 4))
    plt.hist(Ytest.flatten(), bins=100, color="orange", alpha=0.5, label="pred")
    plt.hist(Yhat.flatten(), bins=100, color="green", alpha=0.5, label="true")
    plt.legend()
    plt.title("value distribution")
    plt.show()


def calc_precision_recall(Ytest, Yhat, threshold=0.002, graph=True):
    lr_precision, lr_recall, _ = precision_recall_curve(Ytest, Yhat)
    try:
        lr_f1 = f1_score(Ytest, (Yhat > threshold).astype(int))

    except:
        traceback.print_exc()
        lr_f1 = 1
    lr_f1, lr_auc = lr_f1, auc(lr_recall, lr_precision)

    if graph:
        pyplot.title("Receiver Operating Characteristic")
        pyplot.plot(
            lr_recall,
            lr_precision,
            "b",
            label="F1 = %0.2f , AUC = %0.2f" % (lr_f1, lr_auc),
        )
        pyplot.legend(loc="lower right")
        pyplot.xlim([0, 1])
        pyplot.ylim([0, 1])
        pyplot.ylabel("Precision")
        pyplot.xlabel("Recall")
        pyplot.show()
    return lr_f1, lr_auc

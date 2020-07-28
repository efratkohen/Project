from ml_prepare import ML_prepare
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns 

def pca_plot(x_partly:pd.DataFrame, color_col=0):  
    pca_model = PCA(n_components=2)
    x_no_nan = x_partly.dropna()

    pca_model.fit(x_no_nan)
    X_2D = pca_model.transform(x_no_nan)
    pca_dict = dict(PCA1=X_2D[:, 0], PCA2=X_2D[:, 1])
    pca_results = pd.DataFrame(pca_dict)
    if color_col==0:
        sns.scatterplot(data=pca_results, x='PCA1', y='PCA2')
    else:
        sns.scatterplot(data=pca_results, x='PCA1', y='PCA2', species=color_col)

def get_x_totals(x):
    total_cols = [col for col in x.columns if 'Total' in col]
    x_totals = x.loc[:,total_cols]
    return x_totals

def get_x_filaments(x):
    filament_cols = [col for col in x.columns if 'Filaments_' in col]
    x_filaments = x.loc[:,filament_cols]
    return x_filaments


if __name__=='__main__':
    data = ML_prepare()
    # data.plot_svi()
    delay = 4
    x, y = data.create_x_y_delayed(days_delay=delay)

    totals_xy = get_x_totals(x)
    filaments_xy = get_x_filaments(x)

    x_totals_clean = x_totals.dropna()
    x_filaments_clean = x_filaments.dropna()










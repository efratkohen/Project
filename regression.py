from ml_prepare import ML_prepare
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.model_selection import train_test_split

def pca_plot(table_xy:pd.DataFrame, section: str, ax_i, color_col='SVI'):  
    x_only = table_xy.loc[:,'x']

    pca_model = PCA(n_components=2)
    pca_model.fit(x_only)

    X_2D = pca_model.transform(x_only)
    pca_dict = dict(PCA1=X_2D[:, 0], PCA2=X_2D[:, 1])
    pca_results = pd.DataFrame(pca_dict)

    color_series = table_xy.loc[:,('y',color_col)].reset_index(drop=True)

    pca_results['color'] = color_series

    g = sns.scatterplot(data=pca_results, x='PCA1', y='PCA2', hue='color', ax=ax_i)
    g.legend_.remove()
    g.set(title = f'{section} colored by {color_col}')
    
def create_section_and_PCA(data: ML_prepare, labled: bool=False):
    section_lst = ['all', 'filaments', 'total_counts', 'various']
    fig, ax = plt.subplots(4,2)
    for i in range(len(section_lst)):
        table_xy = data.get_partial_table(x_section=section_lst[i], y_labels=labled)
        y_cols = table_xy.loc[:,'y'].columns.tolist()
        for j in range(2):
            pca_plot(table_xy, color_col=y_cols[j], section = section_lst[i], ax_i = ax[i,j])
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.suptitle(f'PCA of groups, colored by output, delay = {data.delay} days', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    delay = 4
    data = ML_prepare(delay= delay)
    
    t_all = data.get_partial_table(x_section='all')
    t_filaments = data.get_partial_table(x_section='filaments')
    t_total = data.get_partial_table(x_section='total_counts')
    t_various = data.get_partial_table(x_section='various')

    
    create_section_and_PCA(data)












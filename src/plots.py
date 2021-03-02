import pickle
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

import os
import sys

p = os.path.abspath('../..')
if p not in sys.path:
    sys.path.append(p)

from src.tfidf import *

def plot_svd_components():
    with open('../models/N-1 TSVD Model.p', 'rb') as fp:
        TSVD_train = pickle.load(fp)

    var_ratios = []
    for i in range(101):
        var_ratios.append(select_n_components(TSVD_train.explained_variance_ratio_, ((i+1)/100)))

    plt.figure(figsize=(15,8))
    ax = sns.lineplot(x=var_ratios, y=list(range(101)))
    ax.set(xlabel='Explained Variance (%)', ylabel='Number of SVD Components')
    plt.show()

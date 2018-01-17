import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.cluster import KMeans
import conf as s
from collections import defaultdict
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

def load_data():
    '''
    This function takes in the PriceBook data and converts it to a pandas DataFrame
    for easy visualization and modeling.
    '''
    df = s.read_and_drop()
    return df

def score_func(scoring_df, eval_df):
    '''
    This function calculates the score of the model, by calculating the profit
    and revenue with the given number of splits.
    '''
    n_revenue = 0
    soln_df = pd.DataFrame()
    for name in eval_df.index.values:
        dfs = scoring_df[scoring_df['AreaName']==name]
        dfs['Label']= eval_df['Labels'][name]
    for lab in eval_df['Labels'].unique():
        dfs = dfs[dfs['Label']==lab]
        dfs['New_Price'] = np.mean(dfs['CurPrice'])
        soln_df = pd.concat([soln_df,dfs], axis=0)
        el = np.log(soln_df['PriceBeta']*soln_df['New_Price'])
        soln_df['unit_sales'] = soln_df['Q'] * (np.exp(-el))
        soln_df['NewRev'] = soln_df['unit_sales'] * soln_df['New_Price']
        n_revenue += np.sum(soln_df['NewRev'])
    return n_revenue, np.sum(scoring_df['CurRev'])

def labels_(eval_df):
    '''
    This model grid searches the features to find the best parameters for
    the KMeans cluster.

    Returns:

    dictionary- Keys = number of clusters, values = labels
    '''
    means_labels = dict()
    for num in range(2,21):
        km = KMeans(n_clusters=num)
        km.fit(eval_df)
        means_labels[num] = km.labels_
    return means_labels

def plotting(scoring_df,eval_df,means_labels):
    '''
    this function helps visualize the revenue curve comparing different grouping of
    areas based on current price of the products.
    '''
    old = []
    new = []
    x = [x for x in range(2,21)]
    for label in means_labels.values():
        eval_df['Labels'] = np.ndarray.tolist(label)
        rev = score_func(scoring_df=scoring_df,
                         eval_df=eval_df)
        # import pdb; pdb.set_trace()
        old.append(rev[1])
        new.append(rev[0])
    ax.plot(x,old, 'b')
    ax.plot(x,new, 'r')
    plt.show()

if __name__ == '__main__':
    df = load_data()
    eval_df = s.cluster_df(df)
    scoring_df = s.scoring_df(df)
    means_labels = labels_(eval_df)
    plotting(scoring_df=scoring_df,eval_df=eval_df, means_labels=means_labels)

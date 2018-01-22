import opt_cluster_eval as f
import conf as s
import numpy as np
import pandas as pd

def table_maker(means_labels):
    '''
    This function takes the means labels and creates the tables to do market basket
    analysis using the apriori algorithm to figure out how likely it would be for
    areas to be grouped together.

    Input:
    means_labels = dictionary, Keys = area name, values = cluster labels over all iterations.

    '''

    basket_df = pd.DataFrame()
    for key, labels in means_labels.items():
        df = pd.DataFrame()
        df['area'] = key
        df['labels'] = labels
        dfs = pd.get_dummies(df['labels'], columns=[x for x in range(1,14)])
        df = df.drop(['labels'], axis =1)
        df = pd.concat([df,dfs],axis=1)
    return basket_df

def cluster_frequency(cluster_lst):
    '''
    This function takes all the Names in the iterations of the model to figure out
    the Areas that are frequently grouped together in a region.

    Input:
    cluster_lst = List of 100 lists containing the 13 clusters with area Names

    Output:
    The count of how many times a certain set of clusters appears over the 100 iterations.


    '''
    formatted_clusters = []
    for clus in cluster_lst:
        y = {tuple(sorted(x)) for x in clus}
        formatted_clusters.append(y)
    for item in formatted_clusters:
        # import pdb; pdb.set_trace()
        print(formatted_clusters[formatted_clusters.count(item) > 27])



if __name__ == '__main__':
    df = s.read_and_drop()
    eval_df = s.cluster_df(df)
    cluster_lab, cluster_lst = f.run_clusters(eval_df)
    cluster_frequency(cluster_lst)
    print(table_maker(means_labels=cluster_lab))

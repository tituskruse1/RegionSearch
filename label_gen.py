'''
Label generator to be used by opt_cluster_eval
'''

# Author: Titus Kruse <Titus.kruse1@gmail.com>

import opt_cluster_eval as f
import conf as s
import pandas as pd

def table_maker(means_labels):
    '''
    This function takes the means labels and creates the tables to do market basket
    analysis using the apriori algorithm to figure out how likely it would be for
    areas to be grouped together.

    Input:
    means_labels = dictionary, Keys = area name, values = cluster labels over all iterations.
    '''
    for key, labels in means_labels.items():
        empty_df = pd.DataFrame()
        empty_df['area'] = key
        empty_df['labels'] = labels
        dfs = pd.get_dummies(empty_df['labels'], columns=[x for x in range(1, 14)])
        empty_df = empty_df.drop(['labels'], axis=1)
        basket_df = pd.concat([empty_df, dfs], axis=1)
    return basket_df


def cluster_frequency(cluster_):
    '''
    This function takes all the Names in the iterations of the model to figure out
    the Areas that are frequently grouped together in a region.

    Input:
    cluster_ = List of 100 lists containing the 13 clusters with area Names

    Output:
    The count of how many times a certain set of clusters appears over the 100 iterations.
    '''
    formatted_clusters = []
    for clus in cluster_:
        area_cluster_names = {tuple(sorted(x)) for x in clus}
        formatted_clusters.append(area_cluster_names)
    for item in formatted_clusters:
        print(formatted_clusters[formatted_clusters.count(item) > 27])


if __name__ == '__main__':
    raw_df = s.read_and_drop()
    eval_df = s.cluster_df(raw_df)
    cluster_lab, cluster_lst = f.run_clusters(eval_df)
    cluster_frequency(cluster_lst)
    print(table_maker(means_labels=cluster_lab))

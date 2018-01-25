'''
generates optimum cluster and returns increase of profit, revenue and
profit_margin.
'''

# Author: Titus Kruse <Titus.kruse1@gmail.com>

from sklearn.cluster import KMeans
import pandas as pd
import conf as s
import initial as f
import numpy as np

def read_and_parce():
    '''
    This function reads in the data and creates the eval_df and scoring_df.

    Inputs:
    None

    Outputs:
    eval_df = DataFrame object
    scoring_df = DataFrame object
    '''
    raw_df = s.read_and_drop()

    return s.scoring_df(raw_df), s.cluster_df(raw_df)


def run_clusters(eval_):
    '''
    This function runs 100 models and stores the areas luster labels dictionary
    as well as creates a list of the names grouped together by cluster and makes
    a list of all iterations.

    Input:
    eval_

    Output:
    cluster_ = dictionary, keys - name of area, values - cluster labels for
    the areas.

    cluster_l= list of areas in each cluster for each iteration.
    '''
    areas = eval_.index.values
    cluster_ = dict()
    cluster_l = []
    for num in range(0, 100):
        model = KMeans(n_clusters=14, n_init=10, max_iter=300)
        model.fit(eval_)
        for name in areas:
            try:
                cluster_[name] += model.labels_[eval_.index == name].tolist()
            except:
                cluster_[name] = model.labels_[eval_.index == name].tolist()
        areas_clustered = []
        for clust in range(0, 14):
            areas_clustered.append(areas[model.labels_ == clust])
        cluster_l.append(areas_clustered)
    return cluster_, cluster_l


def calc_func(cluster_):
    '''
    This function takes in the cluster label dictionary and gets the name of the
    intersections with the other areas when they are grouped in the same cluster.

    Input:
    cluster_ = dictionary, keys - name of area, values - cluster labels for
    the areas.

    Output:
    print out of the names in each of the clusters.
    '''
    names = []
    intersections = []
    for name, counter in cluster_.items():
        names.append(name)
        for name1, counter1 in cluster_.items():
            inter = []
            for ind, val in enumerate(counter):
                if counter1[ind] == counter[ind]:
                    inter.append(val)
                else:
                    pass
            intersections.append((name, name1, len(inter)))
    inter_df = pd.DataFrame(intersections)
    df2 = inter_df[inter_df[2] > 0]

    clusters = dict()
    cluster_legend = dict()
    for name in df2[1].unique():
        dfs = df2[df2[1] == name]
        clusters[name] = dfs[0][dfs[2] > 80].tolist()

    for item in clusters.keys():
        cluster_legend[item] = []

    for val in clusters.values():
        for num in range(1, 14):
            for area in val:
                if type(cluster_legend[area]) == type(1):
                    break
                else:
                    cluster_legend[area] = num
    return None

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
    clusters = []
    for item in formatted_clusters:
        clusters.append(formatted_clusters.count(item))
    max_ = np.argmax(clusters)
    return formatted_clusters[max_], formatted_clusters.count(formatted_clusters[max_])

def best_math(clusters, scoring_):
    '''
    This function will set the regions equal to the optimal clusters and return
    the increase in profit, revenue and profit margin.

    Input:
    opt_cluster

    Output:
    increase in profit, revenue and margin given the optimum clusters.
    '''
    opt_prof = 0
    opt_rev = 0
    mar_lst = []
    for tup in clusters:
        area_df = scoring_[scoring_['AreaName'].isin(tup)]
        new_price = area_df.groupby('ProductId').mean()['CurPrice']
        j_df = area_df.join(new_price, on='ProductId', rsuffix='NewPrice')
        revenue, profit = f.calc_func(j_df)
        opt_prof += profit
        opt_rev += revenue
        mar_lst.append(profit/ revenue)
    print('Current Revenue: ' + str(scoring_['CurRev'].sum()))
    print('Optimum revenue: ' + str(opt_rev))
    print('Current profit: ' + str(scoring_['CurProfit'].sum()))
    print('Optimum profit: ' + str(opt_prof))
    print('Average Margin over optimum prices : ' + str(np.mean(mar_lst)))
    return None


if __name__ == '__main__':
    scoring_df, eval_df = read_and_parce()
    cluster_lab, cluster_lst = run_clusters(eval_df)
    opt_cluster, freq = cluster_frequency(cluster_lst)
    print(best_math(opt_cluster, scoring_df))

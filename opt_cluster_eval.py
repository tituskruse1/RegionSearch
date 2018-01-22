import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict,Counter
import conf as s

def read_and_parce():
    df = s.read_and_drop()
    eval_df = s.cluster_df(df)
    scoring_df = s.scoring_df(df)

    return scoring_df, eval_df

def run_clusters(eval_df):

    areas = eval_df.index.values
    cluster_lab=dict()
    cluster_lst = []
    for num in range(0,100):
        km = KMeans(n_clusters=13, n_init= 10, max_iter=300)
        km.fit(eval_df)
        for idx, name in enumerate(areas):
            try:
                cluster_lab[name] += km.labels_[eval_df.index == name].tolist()
            except:
                cluster_lab[name] = km.labels_[eval_df.index == name].tolist()
        areas_clustered = []
        for clust in range(0,13):
            areas_clustered.append(areas[km.labels_==clust])
        cluster_lst.append(areas_clustered)
    # import pdb; pdb.set_trace()
    return cluster_lab, cluster_lst

def calc_func(cluster_lab):
    names = []
    intersections = []
    for name,counter in cluster_lab.items():
        names.append(name)
        for name1,counter1 in cluster_lab.items():
            inter=[]
            for ind, val in enumerate(counter):
                if counter1[ind] == counter[ind]:
                    inter.append(val)
                else:
                    pass
            intersections.append((name, name1, len(inter)))
    df = pd.DataFrame(intersections)
    df2 = df[df[2]>0]

    clusters = dict()
    cluster_legend = dict()
    for name in df2[1].unique():
        dfs = df2[df2[1]==name]
        clusters[name] = dfs[0][dfs[2] > 80].tolist()

    for item in clusters.keys():
        cluster_legend[item] = []

    for key, val in clusters.items():
        for num in range(1,14):
            for area in val:
                if type(cluster_legend[area]) == type(1):
                    break
                else:
                    cluster_legend[area] = num
    print(cluster_legend)
    return None


if __name__ == '__main__':
    scoring_df, eval_df = read_and_parce()
    cluster_lab, cluster_lst = run_clusters(eval_df)
    print(calc_func(means_labels=means_labels))

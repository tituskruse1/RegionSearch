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
    label_dict = dict()
    pricing_dict=dict()

    for label in eval_df['Labels'].unique():
        temp_df = eval_df[eval_df['Labels']==label]
        label_dict[label] = temp_df.index.values

    for cluster_num, areas in label_dict.items():
        if areas.any():
            df = pd.DataFrame()
            region_price_dict = dict()
            df = scoring_df[scoring_df['AreaName'].isin(areas)]

            new_price = df.groupby('ProductId').mean()['CurPrice']
            j_df = df.join(new_price,on='ProductId',rsuffix='NewPrice')

        else:
            continue
        try:
            el = j_df['FcstBeta'] * -j_df['CurPriceNewPrice']
            j_df['unit_sales'] =j_df['Q'] * np.exp(el)
            j_df['NewRev'] = j_df['unit_sales'] * j_df['CurPriceNewPrice']
            n_revenue += np.sum(j_df['NewRev'])
        except KeyError:

            import pdb; pdb.set_trace()

    return n_revenue, np.sum(scoring_df['CurRev'])

def pricing_function(scoring_df, eval_df):
    '''
    This function generates the new price of a pricing region for each product
    by taking the mean of the current prices.

    '''
    label_dict = defaultdict(list)
    pricing_dict=dict()

    for label in eval_df['Labels'].unique():
        temp_df = eval_df[eval_df['Labels']==label]
        label_dict[label].append(temp_df.index.values)

    #Iterate over clusters
    # for item in label_dict.keys():
    #     df = pd.DataFrame()
    #     region_price_dict = dict()
    #     #Iterate over states in cluster
    #     for name in label_dict[item]:
    #         for area in np.ndarray.tolist(name):
    #             dfs = scoring_df[scoring_df['AreaName']==area]
    #             df = pd.concat([df,dfs],axis = 0)
    #     for prod in df['ProductId'].unique():
    #         df2 = df[df['ProductId']==prod]
    #         region_price_dict[prod] = df2['CurPrice'].mean()
    #         pricing_dict[item] = region_price_dict

    for cluster_num, areas in label_dict.items():
        if areas:
            df = pd.DataFrame()
            region_price_dict = dict()
            df = scoring_df[scoring_df['AreaName'].isin(areas[0])]


            df.groupby('ProductId').mean()['CurPrice']

            for prod in df['ProductId'].unique():
                df2 = df[df['ProductId']==prod]
                region_price_dict[prod] = df2['CurPrice'].mean()
                pricing_dict[cluster_num] = region_price_dict
        else:
            pricing_dict[cluster_num] = None
    return pricing_dict

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
    ax.plot(x,old, 'b',label='Old Revenue')
    ax.plot(x,new, 'r', label='New Revenue')
    ax.set_ylabel('Revenue in $M')
    ax.set_xlabel('# Pricing Regions')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    df = load_data()
    eval_df = s.cluster_df(df)
    scoring_df = s.scoring_df(df)
    means_labels = labels_(eval_df)
    plotting(scoring_df=scoring_df,eval_df=eval_df, means_labels=means_labels)

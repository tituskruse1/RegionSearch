'''
Regional Price analysis algorithms
'''

# Author: Titus Kruse <Titus.kruse1@gmail.com>

import pdb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import conf as s
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    '''
    This function takes in the PriceBook data and converts it to a pandas DataFrame
    for easy visualization and modeling.

    Input:
    None

    Output:
    DataFrame object

    '''
    return s.read_and_drop()


def labels_(eval_):
    '''
    This model grid searches the features to find the best parameters for
    the KMeans cluster.

    Input:
    eval_ - DataFrame object with AreaName as indices, product id as columns
    and Current price as values.

    Output:
    dictionary- Keys = number of clusters, values = labels
    '''
    cluster_labels = dict()
    dists = []
    for num in range(2, 21):
        model = KMeans(n_clusters=num)
        model.fit(eval_)
        cluster_labels[num] = model.labels_
        dists.append(model.inertia_)
    return cluster_labels, dists


def plotting(scoring_, eval_):
    '''
    This function helps visualize the revenue, profit and margin curves comparing
    different grouping of areas based on current price of the products.

    Input:
    scoring_ - DataFrame object with columns 'ProductId','AreaName','CurPrice',
    'CurRev','Q','FcstBeta','CurRev','Cost'

    eval_ - DataFrame object with AreaName as indices, product id as columns
    and Current price as values.

    means_labels- dictionary, keys = number of clusters, values = labels

    Output:
    3 saved figures=
    'Profit_box.jpg'- box plot of density of profit over 50 iterations
    on each number of clusters.
    'Margin_box.jpg'-box plot of density of Profit Margin over 50 iterations
    on each number of clusters.
    'Rev_box.jpg'- box plot of density of Profit Revenue over 50 iterations
    on each number of clusters.

    1 unsaved figure=
    'Revenue Comparison Over Number of Regions'- compares old and new revenue
    of last model ran over number of clusters.
    '''
    fig, ax = plt.subplots()
    # x_vals = [x for x in range(2, 21)]
    rev_box = []
    margin_box = []
    profit_box = []
    distances = []
    for num in range(1, 51):
        old_rev, new_rev, new_prof, new_m, dists = plotting_helper(scoring_=scoring_,
                                                                   eval_=eval_)
        rev_box.append(new_rev)
        margin_box.append(new_m)
        profit_box.append(new_prof)
        distances.append(dists)

    ax = sns.boxplot(pd.DataFrame(distances))
    ax.set_title('Inertia over clusters')
    plt.savefig('5_product_distance_box.jpg')

    plt.show()

    ax = sns.boxplot(pd.DataFrame(profit_box))
    ax.set_title('Profit Curve Over Number of Regions')
    plt.savefig('5_product_Profit_box.jpg')

    plt.show()

    ax = sns.boxplot(pd.DataFrame(margin_box))
    ax.set_title('Profit Margin Over Number of Regions')
    plt.savefig('5_product_Margin_box.jpg')

    plt.show()

    ax = sns.boxplot(pd.DataFrame(rev_box))
    ax.set_title('Density of rev val over 50 iters')
    plt.savefig('5_product_Rev_box.jpg')

    plt.show()


def helper_dict(eval_):
    '''
    This function helps the scoring function by creating a dictionary of
    splits and area cluster labels.

    Input:
    eval_

    Output:
    dictionary = keys - number of splits, values- cluster labels for areas.
    '''
    label_dict = dict()
    for label in eval_['Labels'].unique():
        temp_df = eval_[eval_['Labels'] == label]
        label_dict[label] = temp_df.index.values
    return label_dict


def helper_df(scoring_, areas):
    '''
    This function creates a DataFrame object that helps the scoring function.

    Input:
    scoring_

    Output:
    DataFrame object = columns -'AreaName','ProductId','CurPrice','NewPrice'
    '''
    area_df = scoring_[scoring_['AreaName'].isin(areas)]
    new_price = area_df.groupby('ProductId').mean()['CurPrice']
    return area_df.join(new_price, on='ProductId', rsuffix='NewPrice')


def plotting_helper(scoring_, eval_):
    '''
    This function takes the labels and gathers the new profit, revenue and
    margin using the scoring_ and eval_.

    Inputs:
    scoring_ = DataFrame object
    eval_ = DataFrame object

    Outputs:
    old_rev= list of old revenue values over number of splits
    new_rev= list of new revenue values over number of splits
    new_prof= list of new profit values over number of splits
    new_margins= list of new profit margins over number of splits
    '''
    labels, dists = labels_(eval_)
    old_rev = []
    new_rev = []
    new_prof = []
    new_margins = []

    for label in labels.values():
        eval_['Labels'] = np.ndarray.tolist(label)
        new_r, old_r, profit, margin = score_func(scoring_=scoring_,
                                                  eval_=eval_)
        # import pdb; pdb.set_trace()
        old_rev.append(old_r)
        new_rev.append(new_r)
        new_prof.append(profit)
        new_margins.append(margin)
    return old_rev, new_rev, new_prof, new_margins, dists


def score_func(scoring_, eval_):
    '''
    This function calculates the score of the model, by calculating the profit
    and revenue with the given number of splits.

    Input:
    scoring_ - DataFrame object with columns 'ProductId','AreaName','CurPrice',
    'CurRev','Q','FcstBeta','CurRev','Cost'

    eval_ - DataFrame object with AreaName as indices, product id as columns
    and Current price as values.

    Output:
    Old Revenue - float, calculated by taking the sum of the 'CurRev' column.
    New Revenue - float, calculated using calc_func function.
    New Profit - float, calculated using calc_func function.
    New Profit Margin - float, calculated by dividing the new profit by the  new revenue.
    '''

    n_revenue = 0
    n_profit = 0
    old_rev = np.sum(scoring_['CurRev'])

    label_dict = helper_dict(eval_=eval_)

    for areas in label_dict.values():
        if areas.any():
            j_df = helper_df(scoring_=scoring_, areas=areas)

        else:
            continue
        try:

            rev, prof = calc_func(j_df=j_df)
            n_revenue += rev
            n_profit += prof
            profit_m = n_profit / n_revenue
        except KeyError:

            pdb.set_trace()
    return n_revenue, old_rev, n_profit, profit_m


def calc_func(j_df):
    '''
    This function calculates profit and revenue given the clusters.

    Input:
    DataFrame object

    Output:
    Revenue- float, calculated using the elasticity formula.
    Profit- float, calculated fy subtracting cost of units from revenue.
    '''
    values = j_df['FcstBeta'] * -j_df['CurPriceNewPrice']
    j_df['unit_sales'] = j_df['Q'] * np.exp(values)
    j_df['NewRev'] = j_df['unit_sales'] * j_df['CurPriceNewPrice']
    revenue = np.sum(j_df['NewRev'])
    profit = revenue - np.sum(j_df['unit_sales'] * j_df['Cost'])
    return revenue, profit

if __name__ == '__main__':
    raw_df = load_data()
    eval_df = s.cluster_df(raw_df)
    scoring_df = s.scoring_df(raw_df)
    plotting(scoring_=scoring_df, eval_=eval_df)

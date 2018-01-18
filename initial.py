import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.cluster import KMeans
import conf as s
from collections import defaultdict
import matplotlib.pyplot as plt

def load_data():
    '''
    This function takes in the PriceBook data and converts it to a pandas DataFrame
    for easy visualization and modeling.

    Input:
    None

    Output:
    DataFrame object

    '''
    df = s.read_and_drop()
    return df

def score_func(scoring_df, eval_df):
    '''
    This function calculates the score of the model, by calculating the profit
    and revenue with the given number of splits.

    Input:
    scoring_df - DataFrame object with columns 'ProductId','AreaName','CurPrice',
    'CurRev','Q','FcstBeta','CurRev','Cost'

    eval_df - DataFrame object with AreaName as indices, product id as columns
    and Current price as values.

    Output:
    Old Revenue - float, calculated by taking the sum of the 'CurRev' column.
    New Revenue - float, calculated using calc_func function.
    New Profit - float, calculated using calc_func function.
    New Profit Margin - float, calculated by dividing the new profit by the  new revenue.
    '''
    n_revenue = 0
    n_profit = 0
    soln_df = pd.DataFrame()
    label_dict = dict()
    pricing_dict=dict()
    old_rev = np.sum(scoring_df['CurRev'])

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

            # n_revenue += calc_func(j_df)
            rev , prof = calc_func(j_df=j_df)
            n_revenue += rev
            n_profit += prof
            profit_m = n_profit / n_revenue
        except KeyError:

            import pdb; pdb.set_trace()

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
    el = j_df['FcstBeta'] * -j_df['CurPriceNewPrice']
    j_df['unit_sales'] =j_df['Q'] * np.exp(el)
    j_df['NewRev'] = j_df['unit_sales'] * j_df['CurPriceNewPrice']
    revenue = np.sum(j_df['NewRev'])
    profit = revenue - np.sum(j_df['unit_sales'] * j_df['Cost'])
    return revenue, profit

def pricing_function(scoring_df, eval_df):
    '''
    This function generates the new price of a pricing region for each product
    by taking the mean of the current prices.

    Input:
    scoring_df - DataFrame object with columns 'ProductId','AreaName','CurPrice',
    'CurRev','Q','FcstBeta','CurRev','Cost'

    eval_df - DataFrame object with AreaName as indices, product id as columns
    and Current price as values.

    Output:
    pricing_dict - dictionary, keys = labels of clusters, values = dictionary where
    keys = product id , keys = new price
    '''
    label_dict = defaultdict(list)
    pricing_dict=dict()

    for label in eval_df['Labels'].unique():
        temp_df = eval_df[eval_df['Labels']==label]
        label_dict[label].append(temp_df.index.values)

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

    Input:
    eval_df - DataFrame object with AreaName as indices, product id as columns
    and Current price as values.

    Output:
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
    This function helps visualize the revenue, profit and margin curves comparing different grouping of
    areas based on current price of the products.

    Input:
    scoring_df - DataFrame object with columns 'ProductId','AreaName','CurPrice',
    'CurRev','Q','FcstBeta','CurRev','Cost'

    eval_df - DataFrame object with AreaName as indices, product id as columns
    and Current price as values.

    means_labels- dictionary, keys = number of clusters, values = labels

    Output:
    2 x 2 plot- upper right plot showing the profit of different 
    '''
    fig,ax = plt.subplots(2,2)
    old = []
    new = []
    prof = []
    margins = []

    x = [x for x in range(2,21)]
    for label in means_labels.values():
        eval_df['Labels'] = np.ndarray.tolist(label)
        new_r, old_r, profit, margin = score_func(scoring_df=scoring_df,
                         eval_df=eval_df)
        # import pdb; pdb.set_trace()
        old.append(old_r)
        new.append(new_r)
        prof.append(profit)
        margins.append(margin)
    ax[0,0].plot(x,old, 'b',label='Old Revenue')
    ax[0,0].plot(x,new, 'r', label='New Revenue')
    ax[0,0].set_ylabel('Revenue in $M')
    ax[0,0].set_xlabel('# Pricing Regions')
    ax[0,0].set_title('Revenue Comparison Over Number of Regions')
    ax[0,0].legend()

    ax[0,1].plot(x,prof,'g',label='New Profit')
    ax[0,1].set_title('Profit Curve Over Number of Regions')
    ax[0,1].legend()

    ax[1,0].plot(x,margins,'g',label='Profit Margin')
    ax[1,0].set_title('Profit MArgin Over Number of Regions')
    ax[1,0].legend()
    plt.show()

if __name__ == '__main__':
    df = load_data()
    eval_df = s.cluster_df(df)
    scoring_df = s.scoring_df(df)
    means_labels = labels_(eval_df)
    plotting(scoring_df=scoring_df,eval_df=eval_df, means_labels=means_labels)

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.cluster import KMeans
import conf as s


def load_data():
    '''
    This function takes in the PriceBook data and converts it to a pandas DataFrame
    for easy visualization and modeling.
    '''
    df = s.read_and_drop()
    return df

def feature_select(df):
    '''
    This function selects pertinent features from the large DF to utilize
    in the clustering algorithm:
    columns : 'ElaPrice', 'LocalStandardDeviation','DnElas', 'Locations'
    '''
    for column in df.columns:
        try:
            df[column] = df[column].str.replace('$','').astype(float)
        except:
            pass
    df2 = df.copy()
    return df2

def weak_model(df2):
    '''
    This function takes in the trimmed Dataframe and fits a crude KMeans
    algorithm to it for a baseline.

    returns:

    Expected Profit and Revenue from making one price zone split.
    '''
    km = KMeans()
    km.fit(df2)
    return km.cluster_centers_()


def score_func():
    '''
    This function calculates the score of the model, by calculating the profit
    and revenue with the given number of splits.
    '''
    
    pass

def search(df2):
    '''
    This model grid searches the features to find the best parameters for
    the KMeans cluster.

    Returns:

    Num_clusters, Profit, Revenue
    '''
    km = KMeans()
    params = {'n_clusters': [2,3,4,5,6,7,8]}

    gs = GridSearchCV(estimator=km,param_grid= params, scoring='score_func',
                      refit='best_params_')
    gs.fit(df2)

    return gs.best_params_

if __name__ == '__main__':
    df = load_data()
    df2 = feature_select(df)
    weak_profit,weak_revenue = weak_model(df2)
    print('The weak model makes $' + weak_profit +' Profit, and $' ,weak_revenue + ' Revenue')
    opt_clusters, opt_profit, opt_revenue = search(df2)
    print('The Best model makes $' + weak_profit +' Profit, and $' ,weak_revenue + ' Revenue')

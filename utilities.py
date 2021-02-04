#!/usr/bin/env python
# coding: utf-8
# Author: <s182244@student.dtu.dk>
# License: MIT License

"""
This module includes utility imports and functions for the project
"Machine Learning applied to Shipbuilding Market Analysis".
"""

#Global imports
import os
import sys
import time
import json
import warnings
from copy import copy
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid

#Set plot parameters
plt.style.use('grayscale') #grayscale ggplot seaborn-darkgrid seaborn-colorblind bmh
plt.rcParams['figure.figsize'] = (15, 5)
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'grey'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 1.5
bbox_fig = [0, 0, 0.975, 1] #figure boundary box
#plt.rcParams['lines.color'] = '#1d1e4b' #DTU blue

#Timestamp function
def Timestamp():
    return time.strftime("%Y-%m-%dT%H%M%SZ")

#Warnings filter
warnings.filterwarnings("ignore")

####################################################################################
#Data utilities
####################################################################################
def load_data(path, printout=True):
    """
    This function loads the time series data.
    
    Parameters
    ----------
    path : str
        The path to the csv file.
        
    Returns
    -------
    df : pandas.core.frame.DataFrame
        The DataFrame containing the data.
    """
    df = pd.read_csv(path).rename(columns={'Unnamed: 0': 'Date'})
    df.Date = pd.to_datetime(df.Date)
    df = df.set_index('Date')
    df = df.set_index(df.index.to_period())
    if printout:
        print('Loaded: %d samples x %d features' %df.shape)
    return df

def inverse_transform(y, ID, columns):    
    pt = joblib.load('data/pt.pkl')
    N = y.shape[0]
    df = pd.DataFrame([], columns=columns)
    df[ID] = y.copy()
    df_inv = pd.DataFrame(pt.inverse_transform(df), columns=columns)
    return df_inv[ID].values.reshape(-1, 1)


####################################################################################
#Visualization
####################################################################################
def plot_pred(y_pred, idX_pred, ID, y_true=None,  idX_true=None, show=True, save=None):
    fig, ax = plt.subplots()
    if y_true is not None:
        df_true = pd.DataFrame(y_true, index=idX_true, columns=['True '+indict[ID]['Name']])
        df_true.plot(ax=ax)
        mae, rae, rmse, r2 = compute_error(y_true, y_pred)
        textstr = 'MAE:  {:.2f}\nRAE:  {:.2f}\nRMSE: {:.2f}\nr2:   {:.2f}'.format(mae, rae, rmse,r2)
        props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        ax.text(0.01, 0.97, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        
        
    df_pred = pd.DataFrame(y_pred, index=idX_pred, columns=['Predicted '+indict[ID]['Name']])
    df_pred.plot(ax=ax)
    ax.set_ylabel(indict[ID]['Unit'])
    ax.legend(loc='upper right')
    fig.tight_layout(rect=bbox_fig)
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    plt.close()


####################################################################################
#Math & stats utilities
####################################################################################
from itertools import chain, combinations, product
def subsets(iterable):
    "subsets([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def tuples(iterable):
    "tuples([[1,2],[3]]) --> (1, 3) (2, 3)"
    return tuple(product(*iterable))

def compute_error(trues, predicted):
    """
    This function takes as input the true and predicted series, 
    then computes and returns the Correlation Coefficient, the Mean Average Error,
    the Relative Absolute Error, the Root Mean Square Error and the r  coefficient.
    """
    #corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(0, 1 - np.sum((trues-predicted)**2) / np.sum((trues - np.mean(trues))**2))
    return mae, rae, rmse, r2 #corr, 



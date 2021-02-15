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

#Indicators index
indicators = pd.read_csv('data/indicators.csv')
indict = indicators.set_index('ID').to_dict(orient='index')

#Set plot parameters
path_fig = '../report/figures/'
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
    
def get_scores(predictions, names, model_type):
    scores = pd.DataFrame(data=['MAE', 'RAE', 'RMSE', 'R2'], columns=['Measures'])
    for name in names:
        mae, rae, rmse, r2 = compute_error(predictions['y_true'].values, predictions[name].values)
        scores[name] = [mae, rae, rmse, r2]
    global path_fig
    scores.to_csv('scores/'+model_type+'-scores.csv', index=False)
    df_to_tabular(scores, path_fig+'forecasting/'+model_type+'-scores.tex', digits=2, align=None)
    return scores
    
def df_to_tabular(df, name, digits=None, align=None):
    N, M = len(df), len(df.columns)
    sep = ' & '
    specialc = ['%', '#', '$', '°']
    if digits is None:
        digits = 2
    if align is None:
        align =  M*'c'
    with open(name, 'w') as file:
        file.write('\\begin{tabular}{' + align + '}\n')
        file.write('\\hlineB{2.5}\n')
        header = ''
        for c in df.columns:
            header += c + sep
        header = header[:-len(sep)]
        header += '\\\ \n'
        for c in specialc:
            header = header.replace(c, '\\'+c)
        file.write(header)
        file.write('\\hline\n')
        for key, s in df.iterrows():
            #line = key + ' & %.2f & %.2f'%tuple(v for v in s) + '\\\ \n'
            line = ''
            for value in s:
                if type(value) == int or type(value) == float:
                    value = np.round(value, digits)
                line += str(value) + sep
            line = line[:-len(sep)]
            line += '\\\ \n'
            line = line.replace('nan', ' ')
            for c in specialc:
                line = line.replace(c, '\\'+ c)
            file.write(line)
        file.write('\\hlineB{2.5}\n')
        file.write('\\end{tabular}')


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



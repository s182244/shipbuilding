#!/usr/bin/env python
# coding: utf-8
# Author: <s182244@student.dtu.dk>
# License: MIT License

"""
This module includes classes of the modelling framework for the project
Machine Learning applied to Shipbuilding Market Analysis.
"""

from utilities import *
ts = load_data('data/ts_std.csv', printout=False)

class Baseline():
    """Sklearn-style baseline estimator 
    as defined in project report.
    """
    def __init__(self, **params):
        pass

    def set_params(self, **params):
        pass

    def get_params(self, deep=True):
        pass
        
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        N, M = X.shape
        y_pred = [y for y in X[-1,:]] #X is assumed to contain only lagged y values
        for n in range(N):
            y_pred.append(np.mean(y_pred[-M:]))
        y_pred = np.array(y_pred[M:]).reshape(-1, 1)
        return y_pred
    
    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

class Selector():
    """Data Selector for Model Wrapper
    """
    def __init__(self, indIDs, **params): #df, 
        self.indIDs = indIDs #List of indicator IDs to select from
        self.params = {
            'w' : 0, #Autoregressive window size
            's' : 0, #Moving average smoothing window size
            'd' : 0, #Differentiation order
        } 
        self.params.update(params)
        
    def set_params(self, **params):
        self.params.update(params)

    def get_params(self, deep=True):
        return self.params.copy()
    
    def fit(self, idX, y=None):
        global ts
        buffer = max(sum(self.params.values()), 0)
        buffer_idX = pd.period_range(start=idX.min()-buffer, end=ts.index.max(), freq='M')
        self.ts_buffer = ts.loc[buffer_idX][self.indIDs].transpose().dropna().transpose().copy()
        #Buffer is used to avoid dimensionality problems between train and test data
        
    def transform(self, idX):
        ts_res = self.ts_buffer.copy()
        
        #Smoothing
        if self.params['s'] > 0:
            ts_res = ts_res.rolling(window=self.params['s'], center=False, min_periods=self.params['s']).mean()
        
        #Differencing
        if self.params['d'] > 0:
            ts_res = ts_res.diff(self.params['d'])
        
        #Lagged features/autoregression
        if self.params['w'] > 0:
            ts_lag = ts_res[[]].copy() 
            for l in range(1, self.params['w']+1): #Minimum lag is 1
                ts_lag = ts_lag.merge(ts_res.shift(l), how='left', left_index=True, right_index=True)
            ts_res = ts_lag.copy()
        
        #Select and return generated features
        ts_res = ts_res.loc[idX].copy()
        return ts_res.values.reshape(-1, ts_res.shape[1])
        
    def fit_transform(self, idX, y=None):
        self.fit(idX)
        return self.transform(idX)      
    
class Model():
    """Sklearn-style model wrapper
    """
    def __init__(self, X_selector=None, y_selector=None, estimator=None):
        self.X_selector = X_selector
        self.y_selector = y_selector
        self.estimator = estimator
        self.params = {}

    def set_params(self, **params):
        self.params.update(params)    
        X_sel_params = {k.replace('X_selector__',''):self.params[k] for k in self.params.keys() if 'X_selector__' in k}
        y_sel_params = {k.replace('y_selector__',''):self.params[k] for k in self.params.keys() if 'y_selector__' in k}
        est_params = {k.replace('estimator__',''):self.params[k] for k in self.params.keys() if 'estimator__' in k}
        self.X_selector.set_params(**X_sel_params)
        self.y_selector.set_params(**y_sel_params)
        self.estimator.set_params(**est_params)

    def get_params(self, deep=True):
        return self.params.copy()
        
    def fit(self, idX, y=None, **fit_params):
        self.params.update(fit_params)
        X = self.X_selector.fit_transform(idX)
        y = self.y_selector.fit_transform(idX)
        self.estimator.fit(X, y)
        
    def predict(self, idX):
        X = self.X_selector.transform(idX)
        return self.estimator.predict(X)
    
    def score(self, idX, y=None, sample_weight=None):
        y_pred = self.predict(idX)
        #print(y_pred.shape)
        y_true = self.y_selector.transform(idX)
        #print(y_true.shape)
        return r2_score(y_true, y_pred, sample_weight=sample_weight)


class Splitter(): 
    """
    Sklearn-style Time Series cross-validator, 
    inspired from `sklearn.model_selection.TimeSeriesSplit`
    
    Note: when passed, tt_split over-rules max_train_size and/or test_size
    """

    def __init__(self, n_splits=5, max_train_size=None, test_size=None, tt_split=None): 
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.tt_split = tt_split 

    def split(self, X=None, y=None, groups=None):       
        n_samples = len(X)
        n_splits = self.n_splits
        if self.tt_split:
            self.max_train_size = int(self.tt_split*n_samples/n_splits)
            self.test_size = int((1-self.tt_split)*n_samples/n_splits)   
        n_folds = n_splits + 1
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        indices = np.arange(n_samples)
        
        ######### Main changes made here
        step = n_samples // (n_folds-1)
        if self.test_size and self.test_size < step:
            test_size = self.test_size
            test_starts = range((n_samples % step) + (step - test_size ), n_samples, step) #HERE
        else:
            test_size = (n_samples // n_folds)
            test_starts = range(test_size + n_samples % n_folds, n_samples, test_size)
        #########
        
        for test_start in test_starts:
            if self.max_train_size and self.max_train_size < test_start:
                yield (indices[test_start - self.max_train_size:test_start],
                       indices[test_start:test_start + test_size])
            else:
                yield (indices[:test_start],
                       indices[test_start:test_start + test_size])
        
    def get_n_splits(self, X=None, y=None, groups=None): #For compatibility
        return self.n_splits
    
    
class CrossValidation():
    """Grid search cross validation for Model wrapper
    """
    def __init__(self, model, grid, splitter, **params):
        self.model = model
        self.grid = ParameterGrid(grid)
        self.splitter = splitter
        self.params = params
    def fit(self, idX, y=None):
        self.cvres = {'param_'+k:[] for k in self.grid[0].keys()}
        for n in range(self.splitter.get_n_splits()):
            self.cvres['split'+str(n)+'_train_score'] = []
            self.cvres['split'+str(n)+'_test_score'] = []

        for params in self.grid:
            for k in params.keys():
                self.cvres['param_'+k].append(params[k])
            self.model.set_params(**params)
            n=0
            for train_split, test_split in self.splitter.split(idX):
                train_idX, test_idX = idX[train_split].copy(), idX[test_split].copy()
                self.model.fit(train_idX)
                self.cvres['split'+str(n)+'_train_score'].append(self.model.score(train_idX))
                self.cvres['split'+str(n)+'_test_score'].append(self.model.score(test_idX))
                n+=1

        self.res = pd.DataFrame(self.cvres)
        self.res['mean_train_score'] = self.res[[c for c in self.res.columns if 'train_score' in c]].mean(axis=1)
        self.res['mean_test_score'] = self.res[[c for c in self.res.columns if 'test_score' in c]].mean(axis=1)
               
    def save_results(self, name, path=None, timestamp=True):
        if timestamp:
            name = Timestamp()+name
        if path is not None:
            name = path+name
            
        self.res.to_csv(name + '.csv', index=False)
        params = {}
        params['splitter'] = {
            'n_splits' : self.splitter.n_splits,
            'max_train_size' : self.splitter.max_train_size,
            'test_size' : self.splitter.test_size,
            'tt_split' : self.splitter.tt_split            
        }
        
        with open(name+'Params.json', 'w') as fp:
            json.dump(params, fp)
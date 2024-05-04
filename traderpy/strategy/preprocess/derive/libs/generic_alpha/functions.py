"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
import xarray as xr
from joblib import wrap_non_picklable_objects

__all__ = ['make_function']
ind_label = pd.DataFrame()

class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity, para):
        self.function = function
        self.name = name
        self.arity = arity
        self.para = para

    def __call__(self, *args):
        return self.function(*args)


def make_function(function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)    
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) == 0.0, np.divide(x1, x2), np.nan)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.sign(x1) >= 0, np.sqrt(x1), np.sign(x1)*np.sqrt(np.abs(x1)))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.sign(x1) >= 0, np.log(x1), np.sign(x1)*np.log(np.abs(x1)))


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    #with np.errstate(divide='ignore', invalid='ignore'):
    return np.divide(1.0, x1)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))
    
def _rank(x1):
    x1 = pd.DataFrame(x1)
    return x1.rank(pct=True).values

def _scale(x1, a=1):
    a = int(a)
    return (a*x1 / np.nansum(np.abs(x1)))

def _signedpower(x1, a=2):
    a = int(a)
    return np.sign(x1) * np.power(np.abs(x1), a)

def _delay(x1,d=5):
    x1 = pd.DataFrame(x1)
    d = 2 if int(d)<=0 else int(d)
    return x1.shift(periods=d,axis=1).values.astype(float)

'''   
def _correlation(x1,x2,d=5):
    return pd.DataFrame(x1).rolling(window=d,axis=1).corr(pd.DataFrame(x2)).\
            replace([-np.inf,np.inf],[-1,1]).values

def _covariance(x1,x2,d=5):
    return pd.DataFrame(x1).rolling(window=d,axis=1).cov(pd.DataFrame(x2)).values
'''    
def _covariance(x1,x2,d=5):
    d = 2 if int(d)<=0 else int(d)
    roll1 = np.array(xr.DataArray(x1, dims=('a', 'b')).rolling(b=d).construct('window_dim').values)
    roll1[:, : d-1, :] = np.nan
    roll2 = np.array(xr.DataArray(x2, dims=('a', 'b')).rolling(b=d).construct('window_dim').values)
    roll2[:, : d-1, :] = np.nan
    cov = (  ( roll1 - np.expand_dims(roll1.mean(axis=2), axis=2) ) *
         ( roll2 - np.expand_dims(roll2.mean(axis=2), axis=2) )  ).sum(axis=2) / d
    return cov.astype(float) 

def _correlation(x1,x2,d=5):
    d = 2 if int(d)<=0 else int(d)
    roll1 = np.array(xr.DataArray(x1, dims=('a', 'b')).rolling(b=d).construct('window_dim').values)
    roll1[:, : d-1, :] = np.nan
    roll2 = np.array(xr.DataArray(x2, dims=('a', 'b')).rolling(b=d).construct('window_dim').values)
    roll2[:, : d-1, :] = np.nan
    cov = (  ( roll1 - np.expand_dims(roll1.mean(axis=2), axis=2) ) *
         ( roll2 - np.expand_dims(roll2.mean(axis=2), axis=2) )  ).sum(axis=2) / d
    corr = cov /  ( roll1.std(axis=2) * roll2.std(axis=2) )
    return corr.astype(float) 

def _delta(x1,d=5):
    d = 2 if int(d)<=0 else int(d)
    return x1.astype(float)-pd.DataFrame(x1).shift(periods=d,axis=1).values.astype(float)

def _decay_linear(x1,d=5):
    d = 2 if int(d)<=0 else int(d)
    roll1 = np.array( xr.DataArray(x1, dims=('a', 'b')).rolling(b=d+1).construct('window_dim') )
    roll1[:, : d , :] = np.nan
    w =2 * np.arange(d, -1, -1)/(d * (d+1))
    res = roll1.dot(w)
    return res.astype(float)
            
def _ts_stddev(X, d=5):
    d = 2 if int(d)<=0 else int(d)
    X = pd.DataFrame(X)
    res = X.rolling(window=d, min_periods=d, axis = 1).std()
    return res.values.astype(float)

'''
def _ts_product(X, d=5):
    X = pd.DataFrame(X)
    res = X.rolling(window=d, min_periods=d, axis = 1).apply(np.prod, raw=True)
    return res.values
'''

def _ts_sum(X, d=5):
    d = 2 if int(d)<=0 else int(d)
    X = pd.DataFrame(X)
    res = X.rolling(window=d, min_periods=d, axis = 1).sum()
    return res.values.astype(float)

'''
def _ts_rank(X, d=5):
    X = pd.DataFrame(X)
    res = pd.DataFrame(index=range(X.shape[0]), columns=range(X.shape[1]))
    for i in range(d - 1, X.shape[1]):
        res[i] = X.iloc[:, i - 4 : i + 1].rank(axis=1).iloc[:, -1]        
    return res.values.astype(float)


def _ts_argmax(X, d=5):
    X = pd.DataFrame(X)
    res = X.rolling(window=d, min_periods=d, axis = 1).apply(np.argmax, raw=True)
    return res.values
    
def _ts_argmin(X, d=5):
    X = pd.DataFrame(X)
    res = X.rolling(window=d, min_periods=d, axis = 1).apply(np.argmin, raw=True)
    return res.values
'''

def _ts_argmax(x1, d=5):
    d = 2 if int(d)<=0 else int(d)
    roll1 = xr.DataArray(x1, dims=('a', 'b')).rolling(b=d).construct('window_dim').fillna(0)
    argmax = roll1.argmax(dim='window_dim')
    res = argmax.values
    res[:, : d-1] = np.nan
    return res.astype(float)

def _ts_argmin(x1, d=5):
    d = 2 if int(d)<=0 else int(d)
    roll1 = xr.DataArray(x1, dims=('a', 'b')).rolling(b=d).construct('window_dim').fillna(0)
    argmin = roll1.argmin(dim='window_dim')
    res = argmin.values
    res[:, : d-1] = np.nan
    return res.astype(float)

def _ts_product(x1, d=5):
    d = 2 if int(d)<=0 else int(d)
    roll1 = xr.DataArray(x1, dims=('a', 'b')).rolling(b=d).construct('window_dim')
    product = roll1.prod(dim='window_dim')
    res = product.values
    res[:, : d-1] = np.nan
    return res.astype(float)

def _ts_rank(x1,d=5):
    d = 2 if int(d)<=0 else int(d)
    roll1 = xr.DataArray(x1, dims=('a', 'b')).rolling(b=d).construct('window_dim')
    rank = roll1.rank(dim='window_dim')/d
    res = rank.values[:,:,-1]
    res[:, : d-1] = np.nan
    return res.astype(float)

def _ts_max(X, d=5):
    d = 2 if int(d)<=0 else int(d)
    X = pd.DataFrame(X)
    res = X.rolling(window=d, min_periods=d, axis = 1).max()
    return res.values.astype(float)
    
def _ts_min(X, d=5):
    d = 2 if int(d)<=0 else int(d)
    X = pd.DataFrame(X)
    res = X.rolling(window=d, min_periods=d, axis = 1).min()
    return res.values.astype(float)

def _ts_mean(X, d=5):
    d = 2 if int(d)<=0 else int(d)
    X = pd.DataFrame(X)
    res = X.rolling(window=d, min_periods=d, axis = 1).mean()
    return res.values.astype(float)

def _ts_sma(x1,n,m):
    if (int(n)<=1 or int(m)<=0):
        n = 5
        m = 1
    elif m/n > 1:
        m = 1
    else:
        n = int(n)
        m = int(m)
    x2=pd.DataFrame(x1.copy())
    return x2.ewm(axis=1,alpha=m/n).mean().values

def _ts_wma(x1,d=5):
    d = 2 if int(d)<=0 else int(d)
    weights = 2 * np.arange(1, d+1) / (d * (d+1))
    x2=x1.copy()
    x2[:,0:d-1]=np.nan
    for i in range(d-1,x1.shape[1]):
        x2[:,i] = x1[:,i-d+1:i+1].dot(weights)
    return x2.astype(float)


def _ifcondition_g(condition_var1, condition_var2, x1, x2):
    flag = pd.DataFrame(condition_var1 > condition_var2)
    A=pd.DataFrame(index=flag.index,columns=flag.columns)
    if type(x1)!=float:
        x1=pd.DataFrame(x1)
    if type(x2)!=float:
        x2=pd.DataFrame(x2)
    A[flag] = x1
    A[~flag] = x2
    return A.values.astype(float)



def _ifcondition_ge(condition_var1, condition_var2, x1, x2):
    flag = pd.DataFrame(condition_var1 >= condition_var2)
    A=pd.DataFrame(index=flag.index,columns=flag.columns)
    if type(x1)!=float:
        x1=pd.DataFrame(x1)
    if type(x2)!=float:
        x2=pd.DataFrame(x2)
    A[flag] = x1
    A[~flag] = x2
    return A.values.astype(float)


def _ifcondition_e(condition_var1, condition_var2, x1, x2):
    flag = pd.DataFrame(condition_var1 == condition_var2)
    A=pd.DataFrame(index=flag.index,columns=flag.columns)
    if type(x1)!=float:
        x1=pd.DataFrame(x1)
    if type(x2)!=float:
        x2=pd.DataFrame(x2)
    A[flag] = x1
    A[~flag] = x2
    return A.values.astype(float)


def _ts_sumif(x1, condition_var1, condition_var2, d=5):
    d = 2 if int(d)<=0 else int(d)
    flag = pd.DataFrame(condition_var1 > condition_var2)
    A=pd.DataFrame(index=flag.index,columns=flag.columns)
    x1 = pd.DataFrame(x1)
    A[flag] = x1
    A[~flag] = 0
    return A.rolling(axis=1,window=d).sum().values.astype(float)


def _ts_count(condition_var1, condition_var2, d=5):
    #condition 只传condition_var1 > condition_var2即可
    d = 2 if int(d)<=0 else int(d)
    condition=pd.DataFrame(condition_var1 > condition_var2)
    return condition.rolling(window=d,axis=1).sum().values.astype(float)


def _ts_highday(x1, d=5):
    d = 2 if int(d)<=0 else int(d)
    roll1 = np.array(xr.DataArray(x1, dims=('a', 'b')).rolling(b=d).construct('window_dim').fillna(-10**8).values)
    roll1[:, : d-1, :] = np.nan
    res = (d - 1 - roll1.argmax(axis=2)).astype(float)
    res[:, : d-1] = np.nan
    return res.astype(float)   

def _ts_lowday(x1, d=5):
    d = 2 if int(d)<=0 else int(d)
    roll1 = np.array(xr.DataArray(x1, dims=('a', 'b')).rolling(b=d).construct('window_dim').fillna(10**8).values)
    roll1[:, : d-1, :] = np.nan
    res = (d - 1 - roll1.argmin(axis=2)).astype(float)
    res[:, : d-1] = np.nan
    return res.astype(float) 

def _indneutral(x1):
    global ind_label
    A1=pd.DataFrame(x1)
    A1.insert(0,"Industry",ind_label.reset_index(drop=True))
    A2 = A1.groupby(["Industry"]).apply(lambda x: x-x.mean())
    return A2.values


add2 = _Function(function=np.add, name='add', arity=2, para=[1,1])
sub2 = _Function(function=np.subtract, name='sub', arity=2, para=[1,1])
mul2 = _Function(function=np.multiply, name='mul', arity=2, para=None)
div2 = _Function(function=np.divide, name='div', arity=2, para=None)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1, para=[1])
log1 = _Function(function=_protected_log, name='log', arity=1, para=[1])
neg1 = _Function(function=np.negative, name='neg', arity=1, para=[1])
inv1 = _Function(function=_protected_inverse, name='inv', arity=1, para=[1])
abs1 = _Function(function=np.abs, name='abs', arity=1, para=[1])
max2 = _Function(function=np.maximum, name='max', arity=2, para=[1,1])
min2 = _Function(function=np.minimum, name='min', arity=2, para=[1,1])
# sin1 = _Function(function=np.sin, name='sin', arity=1)
# cos1 = _Function(function=np.cos, name='cos', arity=1)
# tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1, para=[1])
rank1 = _Function(function=_rank, name='rank', arity=1, para=[1])
scale1 = _Function(function=_scale, name='scale', arity=1, para=[1])
signedpower1 = _Function(function=_signedpower, name='signedpower', arity=1, para=[1])
delay2 = _Function(function=_delay, name='delay', arity=2, para=[0,1])
corr3 = _Function(function=_correlation, name='correlation', arity=3, para=[0,1,1])
cov3 = _Function(function=_covariance, name='covariance', arity=3, para=[0,1,1])
delta2 = _Function(function=_delta, name='delta', arity=2, para=[0,1])
decay_linear2 = _Function(function=_decay_linear, name='decay_linear', arity=2, para=[0,1])
ts_min2 = _Function(function=_ts_min, name='ts_min', arity=2, para=[0,1])
ts_max2 = _Function(function=_ts_max, name='ts_max', arity=2, para=[0,1])
ts_argmin2 = _Function(function=_ts_argmin, name='ts_argmin', arity=2, para=[0,1])
ts_argmax2 = _Function(function=_ts_argmax, name='ts_argmax', arity=2, para=[0,1])
ts_rank2 = _Function(function=_ts_rank, name='ts_rank', arity=2, para=[0,1])
ts_sum2 = _Function(function=_ts_sum, name='ts_sum', arity=2, para=[0,1])
ts_mean2 = _Function(function=_ts_mean, name='ts_mean', arity=2, para=[0,1])
ts_product2 = _Function(function=_ts_product, name='ts_product', arity=2, para=[0,1])
ts_stddev2 = _Function(function=_ts_stddev, name='ts_stddev', arity=2, para=[0,1])
ts_sma3 = _Function(function=_ts_sma, name='ts_sma', arity=3, para=[0,0,1])
ts_wma2 = _Function(function=_ts_wma, name='ts_wma', arity=2, para=[0,1])
sign1 = _Function(function=np.sign, name='sign', arity=1, para=[1])
power2 = _Function(function=np.power, name='power', arity=2, para=[1,1])
ifcondition_g4 = _Function(function=_ifcondition_g, name='ifcondition_g', arity=4, para=[1,1,1,1])
ifcondition_ge4 = _Function(function=_ifcondition_ge, name='ifcondition_ge', arity=4, para=[1,1,1,1])
ifcondition_e4 = _Function(function=_ifcondition_e, name='ifcondition_e', arity=4, para=[1,1,1,1])
ts_sumif4 = _Function(function=_ts_sumif, name='ts_sumif', arity=4, para=[0,1,1,1])
ts_count3 = _Function(function=_ts_count, name='ts_count', arity=3, para=[0,1,1])
ts_highday2 = _Function(function=_ts_highday, name='ts_highday', arity=2, para=[0,1])
ts_lowday2 = _Function(function=_ts_lowday, name='ts_lowday', arity=2, para=[0,1])
indneutral1 = _Function(function=_indneutral, name='indneutral', arity=1, para=[1])


_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1, 
                 'max': max2,
                 'min': min2,
                 #'sin': sin1,
                 #'cos': cos1,
                 #'tan': tan1,
                 #'sig': sig1,
                 'rank': rank1,
                 'scale': scale1,
                 'signedpower': signedpower1,
                 'delay': delay2,
                 'correlation': corr3,
                 'covariance': cov3,
                 'delta': delta2,
                 'decay_linear': decay_linear2,
                 'ts_min': ts_min2,
                 'ts_max': ts_max2,
                 'ts_argmin': ts_argmin2,
                 'ts_argmax': ts_argmax2,
                 'ts_rank': ts_rank2,
                 'ts_sum': ts_sum2,
                 'ts_mean': ts_mean2,
                 'ts_product': ts_product2,
                 'ts_stddev': ts_stddev2,
                 'ts_sma': ts_sma3,
                 'ts_wma': ts_wma2,
                 'sign': sign1,
                 'power': power2,
                 'ifcondition_g': ifcondition_g4,
                 'ifcondition_e': ifcondition_e4,
                 'ifcondition_ge': ifcondition_ge4,
                 'ts_sumif': ts_sumif4,
                 'ts_count': ts_count3,
                 'ts_highday': ts_highday2,
                 'ts_lowday': ts_lowday2,
                 'indneutral': indneutral1
                 }

function_weights = {add2 : 4,
                 sub2 : 4,
                 mul2 : 4,
                 div2 : 4,
                 sqrt1 : 1,
                 log1 : 1,
                 abs1 : 1,
                 neg1 : 1,
                 inv1 : 1, 
                 max2 : 3,
                 min2 : 3,
                 #'sin': sin1,
                 #'cos': cos1,
                 #'tan': tan1,
                 #'sig': sig1,
                 rank1 : 3,
                 scale1 : 1,
                 signedpower1 : 1,
                 delay2 : 3,
                 corr3 : 6,
                 cov3 : 4,
                 delta2 : 3,
                 decay_linear2 : 1,
                 ts_min2 : 1,
                 ts_max2 : 1,
                 ts_argmin2: 2,
                 ts_argmax2: 2,
                 ts_rank2 : 1,
                 ts_sum2 : 1,
                 ts_mean2 : 1,
                 ts_product2 : 1,
                 ts_stddev2 : 1,
                 ts_sma3 : 1,
                 ts_wma2 : 1,
                 sign1 : 0,
                 power2 : 1,
                 ifcondition_g4 : 4,
                 ifcondition_e4 : 4,
                 ifcondition_ge4 : 4,
                 ts_sumif4 : 2,
                 ts_count3 : 2,
                 ts_highday2 : 1,
                 ts_lowday2 : 1,
                 indneutral1 : 6
                 }


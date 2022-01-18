import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as ss_mvn
from mlwpy import plot_separator
import collections as co
import itertools as it

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (check_X_y, 
                                      check_array, 
                                      check_is_fitted)

# import sklearn.linear_model as linear_model
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm

def plot_boundary(data, tgt, dims, 
                  model, 
                  show_data=False,
                  grid_step = .01,
                  limits=None,
                  ax=None):
    ax = ax if ax else plt.gca()

    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # grab a 2D view of the data and get limits
    twoD = data[:, list(dims)]
    if not limits:
        min_x1, min_x2 = np.min(twoD, axis=0) + 2 * grid_step
        max_x1, max_x2 = np.max(twoD, axis=0) - grid_step
    else:
        min_x1, min_x2 = limits[0]
        max_x1, max_x2 = limits[1]

    if show_data:
        ax.scatter(*twoD.T, c=tgt, cmap=plt.cm.coolwarm, alpha=.3)
        
    # make a grid of points and predict at them
    xs, ys = np.mgrid[min_x1:max_x1:grid_step,
                      min_x2:max_x2:grid_step]
    grid_points = np.c_[xs.ravel(), ys.ravel()]
    # warning:  non-cv fit
    mod_fit = model.fit(twoD, tgt)
    preds = (mod_fit.predict(grid_points)
                    .reshape(xs.shape))

    # plot the predictions at the grid points
    # also cmap=plt.cm.Set1, plt.cm.tab10/20
    ax.pcolormesh(xs,ys,preds,shading='auto',
                      cmap=plt.cm.coolwarm, alpha=.3)
    ax.set_xlim(min_x1, max_x1)#-grid_step)
    ax.set_ylim(min_x2, max_x2)#-grid_step)

    return mod_fit


def plot_contours(data, tgt, dims, 
                   model, 
                   show_data=False,
                   grid_step = .01,
                   limits=None,
                   ax=None):
    ax = ax if ax else plt.gca()
    fig = ax.figure
    
    # grab a 2D view of the data and get limits
    twoD = data[:, list(dims)]
    if not limits:
        min_x1, min_x2 = np.min(twoD, axis=0) + 2 * grid_step
        max_x1, max_x2 = np.max(twoD, axis=0) - grid_step
    else:
        min_x1, min_x2 = limits[0]
        max_x1, max_x2 = limits[1]

    if show_data:
        ax.scatter(*twoD.T, c=tgt, cmap=plt.cm.coolwarm, alpha=.3)
        
    # make a grid of points and predict at them
    xs, ys = np.mgrid[min_x1:max_x1:grid_step,
                      min_x2:max_x2:grid_step]
    grid_points = np.c_[xs.ravel(), ys.ravel()]
    # warning:  non-cv fit
    preds = model.fit(twoD, tgt).predict_proba(grid_points)[:,1].reshape(xs.shape)

    # plot the predictions at the grid points
    cs = ax.contourf(xs, ys, preds, alpha=.3, cmap=plt.cm.coolwarm, vmin=0, vmax=1)
    fig.colorbar(cs)
    
    ax.set_xlim(min_x1, max_x1)#-grid_step)
    ax.set_ylim(min_x2, max_x2)#-grid_step)


def do_linear_svc_separators(svc_maker, ftrs, tgt,
                                 pname, params, ax):
    'create svc(params) and draw seperation boundary'
    xys = (np.linspace(2,8,100),
           np.linspace(2,8,100))

    for p in params:
        kwargs = {pname:p, 'kernel':'linear'}
        svc = svc_maker(**kwargs).fit(ftrs, tgt)
        # plot_separator is in mlwpy.py
        plot_separator(svc, *xys, 
                       '{}={:g}'.format(pname, p), ax=ax)
    
def make_xor():
    xor_data = [[0,0,0],
                [0,1,1],
                [1,0,1],
                [1,1,0]]
    xor_df = pd.DataFrame(xor_data, 
                          columns=['x1', 'x2', 'tgt'])
    return xor_df


def cat_dog_no_con():
    cat_cov=[[1,0],
             [0,9]]
    dog_cov=[[9,0],
             [0,1]]
    return cat_cov, dog_cov

def cat_dog_with_con():
    cat_cov = [[1,2],
               [2,9]]
    dog_cov = [[ 9,.8],
               [.8,1]]
    return cat_cov, dog_cov

def cat_dog_nice_sep():
    cat_cov = [[4,2],
               [2,2]]
    dog_cov = [[6,2],
               [2,3]]
    return cat_cov, dog_cov

def cat_dog_overlap():
    cat_cov = [[15,2],
               [2,2]]
    dog_cov = [[15,2],
               [2,3]]
    return cat_cov, dog_cov

def cat_dog_same_cov():
    cat_cov = [[4,2],
               [2,2]]
    dog_cov = [[4,2],
               [2,2]]
    return cat_cov, dog_cov
    

def make_cats_and_dogs(cat_cov, dog_cov, size=1000):
    cats = ss_mvn(mean=[5,  5], cov=cat_cov).rvs(size)
    dogs = ss_mvn(mean=[15, 5], cov=dog_cov).rvs(size)

    ftrs = np.vstack([cats, dogs])
    cd_df = pd.DataFrame(ftrs, columns=['height', 'weight'])
    cd_df['species'] = np.repeat(['cat', 'dog'], size)

    return cd_df

def make_polo_df():
    # tail_probs = [0.0, .001, .01, .05, .10, .25, 1.0/3.0]
    tail_probs = [0.0, .01, .05, .10, .25]

    lwr_probs = np.array(tail_probs)
    upr_probs = 1-lwr_probs[::-1]
    cent_prob = np.array([.5])

    probs = np.concatenate([lwr_probs, cent_prob, upr_probs])

    # much better than geterr/seterr/seterr
    with np.errstate(divide='ignore'):
        odds     = probs / (1-probs)
        log_odds = np.log(odds)

    index=["{:4.1f}%".format(p) for p in np.round(probs,3)*100]

    polo_dict = co.OrderedDict([("Prob(E)",       probs), 
                                ("Odds(E:not E)", odds), 
                                ("Log-Odds",      log_odds)])
    polo_df = pd.DataFrame(polo_dict, index=index)
    polo_df.index.name="Pct(%)"
    return polo_df


def draw_cov_rectangles():
    # color coding
    # -inf -> 0; 0 -> .5; inf -> 1
    # slowly at the tails; quickly in the middle (near 0)
    def sigmoid(x):  
        return np.exp(-np.logaddexp(0, -x))

    # to get the colors we need, we have to build a raw array
    # with the correct values.  we are really "drawing"
    # inside a numpy array, not on the screen
    def draw_rectangle(arr, pt1, pt2):
        (x1,y1),(x2,y2) = pt1,pt2
        delta_x, delta_y = x2-x1, y2-y1
        r,c = min(y1,y2), min(x1,x2)  # x,y -> r,c
        # assign +/- 1 to each block in the rectangle.  
        # total summation value equals area of rectangle (signed for up/down)
        arr[r:r+abs(delta_y), 
            c:c+abs(delta_x)] += np.sign(delta_x * delta_y)

    # our data points:
    pts = [(1,1), (3,6), (6,3)]
    pt_array = np.array(pts, dtype=np.float64)

    # the array we are "drawing" on:
    draw_arr = np.zeros((10,10))
    ct = len(pts)
    c_magic = 1 / ct**2 # without double counting

    # we use the clever, don't double count method
    for pt1, pt2 in it.combinations(pts, 2):
        draw_rectangle(draw_arr, pt1, pt2)
    draw_arr *= c_magic

    # display the array we drew
    from matplotlib import cm
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.matshow(sigmoid(draw_arr), origin='lower',
        cmap=cm.bwr, vmin=0, vmax=1)
    fig.tight_layout()

    # show a diagonal across each rectangles
    # the array elements are centered in each grid square
    ax.plot([ .5, 2.5],[ .5, 5.5], 'r')  # from 1,1 to 3,6 
    ax.plot([ .5, 5.5],[ .5, 2.5], 'r')  # from 1,1 to 6,3
    ax.plot([2.5, 5.5],[5.5, 2.5], 'b');  # from 3,6 to 6,3


class PiecewiseConstantRegression(BaseEstimator, RegressorMixin):
    def __init__(self, cut_points=None):
        self.cut_points = cut_points

    def fit(self, X, y):
        X, y = check_X_y(X,y)

        recoded_X = self._recode(X)
        # even though the _inner_ model is fit without an intercept
        # our piecewise model *does* have an constant term (but see notes)
        self.coeffs_ = (linear_model.
                             LinearRegression(fit_intercept=False)
                            .fit(recoded_X, y)
                            .coef_)
    def _recode(self, X):
        cp = self.cut_points
        n_pieces = len(cp) + 1
        recoded_X = np.eye(n_pieces)[np.searchsorted(cp, X.flat)]
        return recoded_X
    
    def predict(self, X):
        check_is_fitted(self, 'coeffs_')
        X = check_array(X) 
        recoded_X = self._recode(X)
        return np.dot(recoded_X, self.coeffs_)


def go_cutpoints(ftrs, tgts, cut_points):
    ' warning hard coded for global ftrs/tgt '
    model = PiecewiseConstantRegression(cut_points=cut_points)
    model.fit(ftrs, tgts)
    preds = model.predict(ftrs)

    print('constants:', model.coeffs_)
    print('rmse:', np.sqrt(metrics.mean_squared_error(tgts, preds)))

    plt.plot(ftrs, preds)
    plt.plot(ftrs, tgts, 'r.')

    plt.vlines(cut_points, 0, 8, 'k');


def show_svr_penalty_noise():
    fig, axes = plt.subplots(2, 3, figsize=(10,5), 
                         sharex=True, sharey=True)

    size=100
    noises = [0.01, 0.1, 1.0]
    Cs = [0.005, 1.0]

    results = []
    for (C, noise), ax in zip(it.product(Cs, noises),
                              axes.flat):
        x = np.random.uniform(0,10,size=size)
        y = x + np.random.normal(scale=noise, size=size)

        svr = svm.SVR(kernel='linear', C=C).fit(x.reshape((size,1)), y)

        sv_dex = np.zeros_like(y, dtype=bool)
        sv_dex[svr.support_] = True

        results.append((C, noise, len(x[sv_dex])))

        ax.plot(x[~sv_dex], y[~sv_dex], 'b.', alpha=.3)
        ax.plot(x[sv_dex], y[sv_dex],   'r.', alpha=.3)
        ax.text(0,10,"#SV={}".format(len(svr.support_)))

    [ax.set_ylabel("C={}".format(C))    for ax, C in zip(axes[:,0], Cs)]
    [ax.set_title("noise={}".format(n)) for ax, n in zip(axes[0,:], noises)]

    fig.tight_layout()
    #pd.DataFrame.from_records(results, columns=['C', 'noise', 'SVs']);


def rms_error(actual, predicted):
    ' root-mean-squared-error function '
    # lesser values are better (a<b ... a better)
    mse = metrics.mean_squared_error(actual, predicted)    
    return np.sqrt(mse)
rms_scorer = metrics.make_scorer(rms_error)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
from sklearn import (metrics,
                     preprocessing as skpre)

pd.options.display.float_format = '{:20,.4f}'.format

def get_model_name(model):
    ' return name of model (class) as a string '
    return str(model.__class__).split('.')[-1][:-2]

def high_school_style(ax):
    ' helper to define an axis to look like a typical school plot '
    ax.spines['left'].set_position(('data', 0.0))
    ax.spines['bottom'].set_position(('data', 0.0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    def make_ticks(lims):
        lwr, upr = sorted(lims) #x/ylims can be inverted in mpl
        lwr = np.round(lwr).astype('int') # can return np objs
        upr = np.round(upr).astype('int')
        if lwr * upr < 0:
            return list(range(lwr, 0)) + list(range(1,upr+1))
        else:
            return list(range(lwr, upr+1))

    import matplotlib.ticker as ticker
    xticks = make_ticks(ax.get_xlim())
    yticks = make_ticks(ax.get_ylim())

    ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))

    ax.set_aspect('equal')

def easy_combo(*arrays):
    ndim = len(arrays)
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)

def hand_and_till_M_statistic(test_tgt, test_probs, weighted=False):
    def auc_helper(truth, probs):
        fpr, tpr, _ = metrics.roc_curve(truth, probs)
        return metrics.auc(fpr, tpr)

    classes   = np.unique(test_tgt)
    n_classes = len(classes)

    indicator = skpre.label_binarize(test_tgt, classes=classes)
    avg_auc_sum = 0.0

    # comparing class i and class j
    for ij in it.combinations(classes, 2):
        # use use sum to act like a logical or
        ij_indicator = indicator[:,ij].sum(axis=1,
                                           dtype=np.bool)

        # slightly ugly, can't broadcast these as indexes
        # use .ix_ to save the day
        ij_probs    = test_probs[np.ix_(ij_indicator, ij)]
        ij_test_tgt = test_tgt[ij_indicator]

        i,j = ij
        auc_ij = auc_helper(ij_test_tgt==i, ij_probs[:,0])
        auc_ji = auc_helper(ij_test_tgt==j, ij_probs[:,1])

        # compared to Hand & Till reference
        # no / 2 ... factor it out since it will cancel
        avg_auc_ij = (auc_ij + auc_ji)

        if weighted:
            avg_auc_ij *= ij_indicator.sum() / len(test_tgt)
        avg_auc_sum += avg_auc_ij

    # compared to Hand & Till reference
    # no * 2 ... factored out above and they cancel
    M = avg_auc_sum / (n_classes * (n_classes-1))
    return M

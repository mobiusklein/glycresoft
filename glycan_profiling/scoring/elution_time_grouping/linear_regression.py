'''The barebone-essentials of weighted ordinary least squares and
a RANSAC-wrapper of it.
'''
import random
from collections import namedtuple

import numpy as np
from scipy import stats


WLSSolution = namedtuple("WLSSolution", [
    'yhat', 'parameters', 'data', 'weights', 'residuals',
    'projection_matrix', 'rss', 'press', 'R2', 'variance_matrix'])

'''A structured container for :func:`weighted_linear_regression_fit`
output.
'''

SMALL_ERROR = 1e-5


def prepare_arrays_for_linear_fit(x, y, w=None):
    """Prepare data for estimating parameter values using the
    weighted ordinary least squares method implemented in :func:`weighted_linear_regerssion_fit`

    Parameters
    ----------
    x : :class:`np.ndarray`
        The data vector or matrix of predictors. Does not contain a common
        intercept.
    y : :class:`np.ndarray`
        The response variable, should have the same outer dimension as x
    w : :class:`np.ndarray`, optional
        The optional weight matrix. If omitted, the identity matrix of appropriate
        shape will be returned.

    Returns
    -------
    X : :class:`np.ndarray`
        The predictors, with a common intercept term added.
    y : :class:`np.ndarray`
        The response variable.
    w : :class:`np.ndarray`
        The weight matrix of X
    """
    X = np.vstack((np.ones(len(x)), np.array(x))).T
    Y = np.array(y)
    if w is None:
        W = np.eye(Y.shape[0])
    else:
        W = np.array(w)
    return X, Y, W


def weighted_linear_regression_fit_ridge(x, y, w=None, alpha=None, prepare=False):
    if prepare:
        x, y, w = prepare_arrays_for_linear_fit(x, y, w)
    elif w is None:
        w = np.eye(y.shape[0])
    if alpha is None:
        alpha = 0.0
    p = x.shape[1]
    V = np.linalg.pinv(x.T.dot(w).dot(x) + np.eye(p) * alpha)
    A = V.dot(x.T.dot(w))
    B = A.dot(y)
    H = x.dot(A)
    yhat = x.dot(B)
    residuals = (y - yhat)
    leave_one_out_error = residuals / (1 - np.diag(H))
    press = (np.diag(w) * leave_one_out_error * leave_one_out_error).sum()
    rss = (np.diag(w) * residuals * residuals).sum()
    tss = (y - y.mean())
    tss = (np.diag(w) * tss * tss).sum()
    return WLSSolution(
        yhat, B, (x, y), w, residuals, H,
        rss, press, 1 - (rss / (tss)), V)


def weighted_linear_regression_fit(x, y, w=None, prepare=False):
    """Fit a linear model using weighted least squares.

    Parameters
    ----------
    x : :class:`np.ndarray`
        The data vector or matrix of predictors
    y : :class:`np.ndarray`
        The response variable, should have the same outer dimension as x
    w : :class:`np.ndarray`, optional
        The optional weight matrix
    prepare : bool, optional
        Whether or not to pass the parameters through :func:`prepare_arrays_for_linear_fit`

    Returns
    -------
    WLSSolution
    """
    if prepare:
        x, y, w = prepare_arrays_for_linear_fit(x, y, w)
    elif w is None:
        w = np.eye(y.shape[0])
    V = np.linalg.pinv(x.T.dot(w).dot(x))
    A = V.dot(x.T.dot(w))
    B = A.dot(y)
    H = x.dot(A)
    yhat = x.dot(B)
    residuals = (y - yhat)
    leave_one_out_error = residuals / (1 - np.diag(H))
    press = (np.diag(w) * leave_one_out_error * leave_one_out_error).sum()
    rss = (np.diag(w) * residuals * residuals).sum()
    tss = (y - y.mean())
    tss = (np.diag(w) * tss * tss).sum()
    return WLSSolution(
        yhat, B, (x, y), w, residuals, H,
        rss, press, 1 - (rss / (tss)), V)


def ransac(x, y, w=None, max_trials=100, regularize_alpha=None):
    '''
    RANSAC Regression, inspired heavily by sklearn's
    much more complex implementation
    '''
    X = x
    residual_threshold = np.median(np.abs(y - np.median(y)))

    if w is None:
        w = np.eye(y)

    def loss(y_true, y_pred):
        return np.abs(y_true - y_pred)

    n_trials = 0
    n_samples = X.shape[0]
    min_samples = X.shape[1] * 5
    if min_samples > X.shape[0]:
        min_samples = X.shape[1] + 1

    if min_samples > X.shape[0]:
        if regularize_alpha is not None:
            return weighted_linear_regression_fit_ridge(X, y, w, regularize_alpha)
        return weighted_linear_regression_fit(X, y, w)

    sample_indices = np.arange(n_samples)

    rng = random.Random(1)

    n_inliers_best = 1
    score_best = -np.inf
    X_inlier_best = None
    y_inlier_best = None
    w_inlier_best = None

    while n_trials < max_trials:
        n_trials += 1
        subset_ix = rng.sample(sample_indices, min_samples)
        X_subset = X[subset_ix]
        y_subset = y[subset_ix]
        w_subset = np.diag(np.diag(w)[subset_ix])

        # fit parameters on random subset of the data
        if regularize_alpha is not None:
            fit = weighted_linear_regression_fit_ridge(
                X_subset, y_subset, w_subset, regularize_alpha)
        else:
            fit = weighted_linear_regression_fit(X_subset, y_subset, w_subset)

        # compute goodness of fit for the fitted parameters with
        # the full dataset
        yhat = np.dot(X, fit.parameters)
        residuals_subset = loss(y, yhat)

        # locate inliers based on residual threshold
        inlier_subset_mask = residuals_subset < residual_threshold
        n_inliers_subset = inlier_subset_mask.sum()

        # determine the quality of the fitted parameters for
        # the inliers using R2
        inlier_subset_ix = sample_indices[inlier_subset_mask]
        X_inlier_subset = X[inlier_subset_ix]
        y_inlier_subset = y[inlier_subset_ix]
        w_inlier_subset = np.diag(np.diag(w)[inlier_subset_ix])
        # w_inlier_best = 1

        yhat_inlier_subset = X_inlier_subset.dot(fit.parameters)
        rss = (w_inlier_subset * np.square(
            y_inlier_subset - yhat_inlier_subset)).sum()
        tss = (w_inlier_subset * np.square(
            y_inlier_subset - y_inlier_subset.mean())).sum()

        score_subset = 1 - (rss / tss)

        # If the number of inliers chosen hasn't improved and the score hasn't
        # improved, don't update the current best
        if n_inliers_subset < n_inliers_best and score_subset < score_best:
            continue

        score_best = score_subset
        X_inlier_best = X_inlier_subset
        y_inlier_best = y_inlier_subset
        w_inlier_best = w_inlier_subset

    if regularize_alpha is None:
        return weighted_linear_regression_fit_ridge(
            X_inlier_best, y_inlier_best, w_inlier_best, regularize_alpha)
    # fit the final best inlier set for the final parameters
    return weighted_linear_regression_fit(
        X_inlier_best, y_inlier_best, w_inlier_best)


def fitted_interval(solution, x0, y0, alpha=0.05):
    n = len(solution.residuals)
    k = len(solution.parameters)
    df = n - k
    sigma2 = solution.rss / df
    X = solution.data[0]
    w = solution.weights

    xtx_inv = np.linalg.pinv(X.T.dot(w).dot(X))
    h = x0.dot(xtx_inv).dot(x0.T)
    if not np.isscalar(h):
        h = np.diag(h)

    error_of_fit = np.sqrt(sigma2 * h)

    t = stats.t.isf(alpha / 2., df)
    width = t * error_of_fit
    return np.stack([y0 - width, y0 + width])


def prediction_interval(solution, x0, y0, alpha=0.05):
    """Calculate the prediction interval around `x0` with response
    `y0` given the `solution`.

    Parameters
    ----------
    solution : :class:`WLSSolution`
        The fitted model
    x0 : :class:`np.ndarray`
        The new predictors
    y0 : float
        The predicted response
    alpha : float, optional
        The prediction interval width. Defaults to 0.05

    Returns
    -------
    :class:`np.ndarray` :
        The lower and upper bound of the prediction interval.
    """
    n = len(solution.residuals)
    k = len(solution.parameters)
    xtx_inv = solution.variance_matrix
    df = n - k
    sigma2 = solution.rss / df
    h = x0.dot(xtx_inv).dot(x0.T)
    if not np.isscalar(h):
        h = np.diag(h)
    error_of_prediction = np.sqrt(sigma2 * (1 + h))

    t = stats.t.isf(alpha / 2., df)
    width = t * error_of_prediction
    return np.stack([y0 - width, y0 + width])

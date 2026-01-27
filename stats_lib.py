import numpy as np
import pandas as pd
from math import sqrt, pi, exp

def get_basic_stats(data):
    stats = {
            'mean': np.mean(data),
            'median': np.median(data),
            'var': np.var(data),
            'std': np.std(data),
            'min': min(data),
            'max': max(data)
    }
    return stats

def get_gauss(x, mean, std):
    t1 = 1/(std*sqrt(2 * pi))
    t2 = exp((-1/2)*((x-mean)/std)**2)
    return t1 * t2

def get_quartiles(data):
    return np.percentile(data, [25, 50, 75])

def get_correlation(x_vals, y_vals):
    n = len(x_vals)
    sum_of_xy = sum([x * y for x, y in zip(x_vals, y_vals)])
    sum_x = sum(x_vals)
    sum_y = sum(y_vals)
    sum_x_squared = sum([x**2 for x in x_vals])
    sum_y_squared = sum([y**2 for y in y_vals])
    
    numerator = n * sum_of_xy - (sum_x * sum_y)
    denominator = sqrt((n * sum_x_squared - sum_x**2) * (n * sum_y_squared - sum_y**2))
    
    r = numerator / denominator
    return r

def get_lobf_lin(x_vals, y_vals):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    A = np.array([
        [np.sum(x*x), np.sum(x)],
        [np.sum(x),   len(x)]
    ], dtype=float)

    C = np.array([np.sum(x*y), np.sum(y)], dtype=float)

    # Solve A * B = C
    a, b = np.linalg.solve(A, C)
    return a, b

def get_lobf_quad(x_vals, y_vals):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    A = np.array([
        [np.sum(x**4), np.sum(x**3), np.sum(x**2)],
        [np.sum(x**3), np.sum(x**2), np.sum(x)],
        [np.sum(x**2), np.sum(x), len(x)]
        ], dtype=float)
    C = np.array([
        np.sum(y * x**2), 
        np.sum(y*x), 
        np.sum(y)
        ], dtype=float)
    a, b, c = np.linalg.solve(A, C)
    return a, b, c

def get_y(x, slope, intercept):
    return slope * float(x) + intercept

def get_residuals(x_vals, y_vals):
    """ 
    Returns a list of residuals, takes in x_vals and y_vals
    """

    slope, intercept = get_lobf_lin(x_vals, y_vals)
    residuals = [y - get_y(x, slope, intercept) for x, y in zip(x_vals, y_vals)]
    return residuals

def get_outliers(x_vals, y_vals, n=2):
    """
    Return (outlier_residuals, outlier_indices) where an outlier is:
      abs(residual - mean_residual) > n * std_residual
    """
    residuals = get_residuals(x_vals, y_vals)
    stats = get_basic_stats(residuals)
    mean_r = stats['mean']
    std_r = stats['std']
    if std_r == 0:
        return [], []
    indices = []
    for i, r in enumerate(residuals):
        if abs(r - mean_r) > n * std_r:
            indices.append(i)
    return indices

def remove_outliers(x_vals, y_vals, n=2):
    """
    Remove points whose residual is more than n * std away from mean residual.
    Accepts pandas Series or array-like. Returns (x_filtered, y_filtered) as pandas Series.
    """
    x = pd.Series(x_vals).reset_index(drop=True)
    y = pd.Series(y_vals).reset_index(drop=True)

    indices = get_outliers(x, y, n=n)
    if not indices:
        return x, y

    mask = pd.Series(True, index=x.index)
    mask.iloc[indices] = False  
    return x[mask].reset_index(drop=True), y[mask].reset_index(drop=True)



def get_rmse(x_vals, y_vals):
    """
    Returns the root mean squared error
    """
    residuals = get_residuals(x_vals, y_vals)
    rmse = sqrt(sum(r**2 for r in residuals) / len(residuals))
    return rmse


def weighted_moving_average(data, weights = [1, 2, 3, 2, 1]):
    w_len = len(weights)
    w_sum = sum(weights)
    result = []

    for start in range(len(data) - w_len + 1):
        window = data[start:(start + w_len)]
        weighted_sum = sum(w * v for w, v in zip(weights, window))
        result.append(weighted_sum / w_sum)

    return result

def weighted_moving_average_2d(x_vals, y_vals, weights = [1, 2, 3, 2, 1]):
    x_vals = weighted_moving_average(x_vals, weights)
    y_vals = weighted_moving_average(y_vals, weights)
    return x_vals, y_vals


def fading_moving_average(data, weights=[0.5, 0.5]):
    w0, w1 = weights
    last_value = data[0]
    result = [last_value]

    for current in data[1:]:
        new_value = last_value * w0 + current * w1
        result.append(new_value)
        last_value = new_value

    return result
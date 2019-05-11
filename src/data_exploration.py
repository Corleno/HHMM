#!/usr/bin/python3
# author: Rui Meng
# date: 05/02/2019

import pickle
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(kernel = "gaussian", bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

# piece-wise linear fitting
from scipy import optimize
def piecewise_linear_2(pars, loc_pars, x):
    y0, y1, k1, k2 = pars
    x0 = loc_pars
    return np.piecewise(x, [x <= x0, x > x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y1-k2*80])

def piecewise_linear_3(pars, loc_pars, x):
    x0, x1 = loc_pars
    y0, y1, y2, k1, k2, k3 = pars
    return np.piecewise(x, [x <= x0, (x > x0) & (x<= x1), x > x1], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y1-k2*x1, lambda x:k3*x + y2 - k3*80])

def piecewise_linear_4(pars, loc_pars, x):
    x0, x1, x2 = loc_pars
    y0, y1, y2, y3, k1, k2, k3, k4 = pars
    return np.piecewise(x, [x <= x0, (x > x0) & (x <= x1), (x > x1) & (x <= x2), x > x2], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y1-k2*x1, lambda x:k3*x + y2-k3*x2, lambda x:k4*x + y3-k4*80])

def piecewise_linear_5(pars, loc_pars, x):
    x0, x1, x2, x3 = loc_pars
    y0, y1, y2, y3, y4, k1, k2, k3, k4, k5 = pars
    return np.piecewise(x, [x <= x0, (x > x0) & (x <= x1), (x > x1) & (x <= x2), (x > x2) & (x <= x3), x > x3], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y1-k2*x1, lambda x:k3*x + y2-k3*x2, lambda x:k4*x + y3-k4*x3, lambda x:k5*x + y4-k5*80])

def piecewise_linear_6(pars, loc_pars, x):
    x0, x1, x2, x3, x4 = loc_pars
    y0, y1, y2, y3, y4, y5, k1, k2, k3, k4, k5, k6 = pars
    return np.piecewise(x, [x <= x0, (x > x0) & (x <= x1), (x > x1) & (x <= x2), (x > x2) & (x <= x3), (x > x3) & (x <= x4), x> x4], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y1-k2*x1, lambda x:k3*x + y2-k3*x2, lambda x:k4*x + y3-k4*x3, lambda x:k5*x + y4-k5*x4, lambda x:k6*x + y5-k6*80 ]) 

def piecewise_constant_2(pars, loc_pars, x):
    x0 = loc_pars
    y0, y1 = pars
    return np.piecewise(x, [x <= x0, x > x0], [lambda x: y0, lambda x: y1])

def piecewise_constant_3(pars, loc_pars, x):
    x0, x1 = loc_pars
    y0, y1, y2 = pars
    return np.piecewise(x, [x <= x0, (x > x0) & (x<= x1), x > x1], [lambda x: y0, lambda x: y1, lambda x: y2])

def piecewise_constant_4(pars, loc_pars, x):
    x0, x1, x2 = loc_pars
    y0, y1, y2, y3 = pars
    return np.piecewise(x, [x <= x0, (x > x0) & (x <= x1), (x > x1) & (x <= x2), x > x2], [lambda x: y0, lambda x: y1, lambda x: y2, lambda x: y3])

def piecewise_constant_5(pars, loc_pars, x):
    x0, x1, x2, x3 = loc_pars
    y0, y1, y2, y3, y4 = pars
    return np.piecewise(x, [x <= x0, (x > x0) & (x <= x1), (x > x1) & (x <= x2), (x > x2) & (x <= x3), x > x3], [lambda x: y0, lambda x: y1, lambda x: y2, lambda x: y3, lambda x: y4])

def piecewise_constant_6(pars, loc_pars, x):
    x0, x1, x2, x3, x4 = loc_pars
    y0, y1, y2, y3, y4, y5 = pars
    return np.piecewise(x, [x <= x0, (x > x0) & (x <= x1), (x > x1) & (x <= x2), (x > x2) & (x <= x3), (x > x3) & (x <= x4), x> x4], [lambda x: y0, lambda x: y1, lambda x: y2, lambda x: y3, lambda x: y4, lambda x: y5]) 


def obj(pars, loc_pars, func, x, y):
    pred_y = func(pars, loc_pars, x)
    sse = np.sum((pred_y - y)**2)
    return sse


if __name__ == "__main__":

    try:
        os.chdir(os.path.dirname(__file__))
    except:
        pass

    with open("../data/data_1000/vectorized_data.pickle", "rb") as res:
        features, objective = pickle.load(res)
 
    # select all hpv+ visiting time
    hpv_pos_age = []
    for patient_features in features:
        for patient_feature in patient_features.T:
            if patient_feature[-2] > 0:
                hpv_pos_age.append(patient_feature[0])
    hpv_pos_age = np.asarray(hpv_pos_age)

    # show the histogram of ages for hpv+
    fig = plt.figure()
    hist = plt.hist(hpv_pos_age, bins=50, density = True)

    densities = hist[0]
    quantiles = np.percentile(a = densities, q = [25, 50, 75])
    print("quantile of densities: {}".format(quantiles))

    # plot fitted density
    x_grid = np.linspace(16, 80, num = 100)
    bandwidth = 0.5
    pdf = kde_sklearn(hpv_pos_age, x_grid, bandwidth = bandwidth)
    plt.plot(x_grid, pdf, linewidth = 2, alpha=0.5, color='r', label = "fitted density")

    # print(pdf)
    max_pdf = np.max(pdf)
    fit_density_range = np.asarray([0, max_pdf])
    
    # thresholds = [0.25* max_pdf, 0.5*max_pdf, 0.75*max_pdf]
    # print("thresholds: {}".format(thresholds))
    # # plot thresholds
    # for thres in thresholds:
    #   plt.axhline(y = thres, color = 'b')

    plt.xlim([16,82])    
    plt.legend()
    plt.savefig("../res/age_histogram_hpv_pos.png")
    plt.close(fig)
    
    # Analysis for constant_linear_2
    # x0 = np.array([0.03, 0.03, 0.03, 0.03, 0, 0, 0, 0])
    x0 = np.zeros(2*4)
    n_iter = 2000
    opt_sse = np.inf
    # for i in range(n_iter):
    #     # randomly choose knots

    #     loc_pars = np.sort(np.random.uniform(low = 16, high = 82, size = 3))
    #     # print((piecewise_constant_2(x0, loc_pars, x_grid)) - (pdf))
    #     print(obj(x0, loc_pars, piecewise_constant_4, x_grid, pdf))
    #     res = optimize.minimize(obj, x0 = x0, args=(loc_pars, piecewise_constant_4, x_grid, pdf))
    #     if res.fun < opt_sse:
    #         opt_sse = res.fun
    #         opt_p = res.x
    #         opt_loc = loc_pars
    
    loc_pars = np.array([25.2, 35.6, 60.6])
    print(loc_pars)
    res = optimize.minimize(obj, x0 = x0, args=(loc_pars, piecewise_linear_4, x_grid, pdf))
    if res.fun < opt_sse:
        opt_sse = res.fun
        opt_p = res.x
        opt_loc = loc_pars

    fig = plt.figure()
    hist = plt.hist(hpv_pos_age, bins=50, density = True)
    opt_location = np.concatenate([[16], opt_loc, [82]])
    plt.plot(x_grid, pdf, linewidth = 2, alpha=0.5, color='r', label = "fitted density", linestyle = '--')
    for i in range(4):
        x_grid_i = np.linspace(opt_location[i], opt_location[i+1], 50)[1:]
        plt.plot(x_grid_i, piecewise_linear_4(opt_p, opt_loc, x_grid_i), label = "segmentation {}".format(str(i+1)))
    plt.xlim([16,82]) 
    plt.xlabel("Age")
    plt.ylabel("Density")   
    plt.legend()
    plt.savefig("../res/age_histogram_hpv_pos_fitted.png")
    plt.close(fig)
    
    print (opt_sse, opt_p, opt_loc)

    
    # # Analysis for piecewise_linear_6
    # # x0 = np.array([0.03, 0.03, 0.03, 0.03, 0, 0, 0, 0])
    # x0 = np.zeros(2*6)
    # n_iter = 2000
    # opt_sse = np.inf
    # for i in range(n_iter):
    #     # randomly choose knots

    #     loc_pars = np.sort(np.random.uniform(low = 16, high = 82, size = 5))
    #     res = optimize.minimize(objective, x0 = x0, args=(loc_pars, piecewise_linear_6, x_grid, pdf))
    #     if res.fun < opt_sse:
    #         opt_sse = res.fun
    #         opt_p = res.x
    #         opt_loc = loc_pars
    # plt.plot(x_grid, piecewise_linear_6(opt_p, opt_loc, x_grid), label = "piecewise_6")
    # print (opt_sse, opt_p, opt_loc)
    
    # Plot SSE against N
    # ns = np.array([2,3,4,5,6])
    # sses = np.array([0.00388, 0.00178, 0.00123, 0.00093, 0.00078])
    # fig = plt.figure()
    # plt.plot(ns, sses)
    # plt.scatter(ns, sses)
    # plt.savefig("../res/sse_vs_n.png")
    # plt.close(fig)


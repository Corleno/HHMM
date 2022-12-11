import pickle
import matplotlib.pyplot as plt
import numpy as np
from lifelines.statistics import logrank_test
from hhmm_application import *
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS


def estimate_KM(durations, event_observed, censor_threshlod=None, verbose=False, return_stats=False):
    if verbose:
        fig = plt.figure()
    if censor_threshlod is not None:
        event_observed[durations > censor_threshlod] = 0
    total_events = durations.shape[0]
    unique_durations = np.unique(durations)
    observed = np.array([(event_observed[durations==duration]==1).sum() for duration in unique_durations])
    censored = np.array([(event_observed[durations==duration]==0).sum() for duration in unique_durations])
    removed = observed + censored
    at_risk = np.concatenate([[total_events], (total_events - np.cumsum(removed))[:-1]])
    ps = 1 - observed/at_risk
    survivals = np.cumprod(ps)

    unique_durations = np.concatenate([[0], unique_durations])
    survivals = np.concatenate([[1], survivals])
    # import pdb; pdb.set_trace()
    if verbose:
        plt.step(unique_durations, survivals, where="post")
        plt.plot(unique_durations, survivals, 'C2+', alpha=0.5)
        plt.tight_layout()
        plt.title("Kaplan-Meier")
        plt.show()
        plt.close(fig)
    if return_stats:
        at_risks = np.concatenate([[total_events], at_risk])
        failures = np.concatenate([[0], observed])
        return unique_durations, survivals, at_risks, failures
    else:
        return unique_durations, survivals


def estimate_probability_of_agreement(during_time1, during_time2, censor_event1, censor_event2, delta=0.05):
    """
    Si_t: time-stamp for ts_i, (ni+1)
    Si_p: survival function for ts_i, (ni+1)
    Si_n: number of at risk, (ni)
    Si_d: number of failures, (ni)
    """

    S1_t, S1_p, S1_n, S1_d = estimate_KM(during_time1, censor_event1, return_stats=True)
    S2_t, S2_p, S2_n, S2_d = estimate_KM(during_time2, censor_event2, return_stats=True)

    # find unique timestamps.
    S0_t = np.unique(np.concatenate([S1_t, S2_t]))
    S0_p = list()
    S0_vars = list()
    thetas = list()
    index1 = 0
    index2 = 0
    index0 = 0
    n1 = S1_t.shape[0]
    n2 = S2_t.shape[0]
    coeff1s = np.cumsum(np.array([S1_d[i]/(S1_n[i]*(S1_n[i]-S1_d[i])) for i in range(n1)]))
    coeff2s = np.cumsum(np.array([S2_d[i]/(S2_n[i]*(S2_n[i]-S2_d[i])) for i in range(n2)]))

    # import pdb; pdb.set_trace()

    for t in S0_t:
        # import pdb; pdb.set_trace()
        while S1_t[index1] < t:
            index1 += 1
            if index1 == n1:
                break
        if index1 == n1:
            index1 -= 1
        elif S1_t[index1] > t:
            index1 -= 1
        while S2_t[index2] < t:
            index2 += 1
            if index2 == n2:
                break
        if index2 == n2:
            index2 -= 1
        elif S2_t[index2] > t:
            index2 -= 1

        s1_p = S1_p[index1]
        s2_p = S2_p[index2]
        s0_p = s2_p - s1_p
        s0_vars = s1_p**2*coeff1s[index1] + s2_p**2*coeff2s[index2]
        S0_p.append(s0_p)
        S0_vars.append(s0_vars)
        theta = norm.cdf((-s0_p+delta)/np.sqrt(s0_vars)) - norm.cdf((-s0_p-delta)/np.sqrt(s0_vars))
        thetas.append(theta)
        index0 += 1

    S0_p = np.array(S0_p)
    S0_vars = np.array(S0_vars)
    thetas = np.array(thetas)
    # import pdb; pdb.set_trace()
    return S0_t, S0_p, S0_vars, thetas


def main(index_range=0, p_threshold=0.2):
    cohort_ranges = [[30, 34], [35, 39], [40, 44], [45, 49], [50, 54], [55, 59], [60, 64], [65, 69]]
    cohort_range = cohort_ranges[index_range]
    censor_threshold = None

    # Load results
    with open("model_res/cohort_range_{}_{}_baseline_vempirical.pickle".format(cohort_range[0], cohort_range[1]),
              "rb") as res:
        baseline_vempirical_df = pickle.load(res)
    with open("model_res/cohort_range_{}_{}_model_p{}.pickle".format(cohort_range[0], cohort_range[1], p_threshold),
              "rb") as res:
        model_df = pickle.load(res)
    # import pdb; pdb.set_trace()

    time, censoring, group = baseline_vempirical_df['time'].to_numpy(), baseline_vempirical_df['censoring'].to_numpy(), \
                             baseline_vempirical_df['group'].to_numpy()
    unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low = estimate_KM(
        time[group == 'low risk'], censoring[group == 'low risk'],
        censor_threshlod=censor_threshold)
    unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high = estimate_KM(
        time[group == 'high risk'], censoring[group == 'high risk'],
        censor_threshlod=censor_threshold)
    if censor_threshold is not None:
        censoring[time > censor_threshold] = 0
    results = logrank_test(time[group == 'low risk'], time[group == 'high risk'],
                           censoring[group == 'low risk'],
                           censoring[group == 'high risk'])
    # results.print_summary()
    print('baseline-vempirical log p_value: ', np.log(results.p_value))
    emp_time_low, emp_time_high, emp_censor_low, emp_censor_high = time[group == 'low risk'], time[group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']

    n_low_vempirical = (group == 'low risk').sum()
    n_high_vempirical = (group == 'high risk').sum()
    # import pdb; pdb.set_trace()

    time, censoring, group = model_df['time'].to_numpy(), model_df['censoring'].to_numpy(), model_df['group'].to_numpy()
    unique_durations_model_low, survivals_model_low = estimate_KM(time[group == 'low risk'],
                                                                censoring[group == 'low risk'],
                                                                censor_threshlod=censor_threshold)
    unique_durations_model_high, survivals_model_high = estimate_KM(time[group == 'high risk'],
                                                                  censoring[group == 'high risk'],
                                                                  censor_threshlod=censor_threshold)
    if censor_threshold is not None:
        censoring[time > censor_threshold] = 0
    results = logrank_test(time[group == 'low risk'], time[group == 'high risk'], censoring[group == 'low risk'],
                           censoring[group == 'high risk'])
    # results.print_summary()
    print('HHMM log p_value: ', np.log(results.p_value))
    hhmm_time_low, hhmm_time_high, hhmm_censor_low, hhmm_censor_high = time[group == 'low risk'], time[group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']

    results = logrank_test(emp_time_low, hhmm_time_low, emp_censor_low, hhmm_censor_low)
    print('low risk: ', np.round(results.p_value,2))
    results = logrank_test(emp_time_high, hhmm_time_high, emp_censor_high, hhmm_censor_high)
    print('high risk: ', np.round(results.p_value,2))

    n_low_HHMM = (group == 'low risk').sum()
    n_high_HHMM = (group == 'high risk').sum()
    # import pdb; pdb.set_trace()

    fig = plt.figure()
    plt.step(unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, where="post",
             label="empirical baseline: no frailty ({})".format(n_low_vempirical))
    plt.step(unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, where="post",
             label="empirical baseline: frailty ({})".format(n_high_vempirical))
    plt.step(unique_durations_model_low, survivals_model_low, where="post",
             label="HCTIHMM: no frailty ({})".format(n_low_HHMM))
    plt.step(unique_durations_model_high, survivals_model_high, where="post",
             label="HCTIHMM: frailty ({})".format(n_high_HHMM))
    plt.xlim(0, 10)
    plt.ylim(0.82, 1.)
    plt.legend()
    plt.tight_layout()
    # plt.title("Kaplan Meier curves for baseline and HCTIHMM model")
    if censor_threshold is None:
        plt.savefig("MC_CR{}_{}_threshold{}.png".format(cohort_range[0], cohort_range[1], p_threshold))
    else:
        plt.xlim(0, censor_threshold)
        plt.savefig(
            "MC_CR{}_{}_threshold{}_c{}.png".format(cohort_range[0], cohort_range[1], p_threshold, censor_threshold))
    plt.show()
    plt.close(fig)

    return 0


def compute_p_value(cohort_range, p_threshold, baseline_vempirical_df, option="nonfrailty", diff_average=False):
    model_df = extract_model_df(cohort_range, p_threshold=p_threshold)
    censor_threshold = None
    time, censoring, group = baseline_vempirical_df['time'].to_numpy(), baseline_vempirical_df['censoring'].to_numpy(), \
                             baseline_vempirical_df['group'].to_numpy()
    # unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low = estimate_KM(
    #     time[group == 'low risk'], censoring[group == 'low risk'],
    #     censor_threshlod=censor_threshold)
    # unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high = estimate_KM(
    #     time[group == 'high risk'], censoring[group == 'high risk'],
    #     censor_threshlod=censor_threshold)
    if censor_threshold is not None:
        censoring[time > censor_threshold] = 0
    emp_time_low, emp_time_high, emp_censor_low, emp_censor_high = time[group == 'low risk'], time[
        group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']

    time, censoring, group = model_df['time'].to_numpy(), model_df['censoring'].to_numpy(), model_df['group'].to_numpy()
    # unique_durations_model_low, survivals_model_low = estimate_KM(time[group == 'low risk'],
    #                                                             censoring[group == 'low risk'],
    #                                                             censor_threshlod=censor_threshold)
    # unique_durations_model_high, survivals_model_high = estimate_KM(time[group == 'high risk'],
    #                                                               censoring[group == 'high risk'],
    #                                                               censor_threshlod=censor_threshold)
    if censor_threshold is not None:
        censoring[time > censor_threshold] = 0
    hhmm_time_low, hhmm_time_high, hhmm_censor_low, hhmm_censor_high = time[group == 'low risk'], time[
        group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']

    if option == "nonfrailty":
        results = logrank_test(emp_time_low, hhmm_time_low, emp_censor_low, hhmm_censor_low)
        print('low risk: ', np.round(results.p_value, 2))
        if diff_average:
            return results.p_value, emp_time_low[emp_censor_low==1].mean() - hhmm_time_low[
                hhmm_censor_low==1].mean()
        else:
            return results.p_value
    if option == "frailty":
        results = logrank_test(emp_time_high, hhmm_time_high, emp_censor_high, hhmm_censor_high)
        print('high risk: ', np.round(results.p_value, 2))
        if diff_average:
            # import pdb; pdb.set_trace()
            return results.p_value, emp_time_high[emp_censor_high == 1].mean() - hhmm_time_high[
                hhmm_censor_high == 1].mean()
        else:
            return results.p_value


def estimate_prob_of_agreement_hhmm(cohort_range, p_threshold, delta=0.05):
    model_df = extract_model_df(cohort_range, p_threshold=p_threshold)
    time, censoring, group = model_df['time'].to_numpy(), model_df['censoring'].to_numpy(), model_df['group'].to_numpy()
    hhmm_time_low, hhmm_time_high, hhmm_censor_low, hhmm_censor_high = time[group == 'low risk'], time[
        group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']
    # import pdb; pdb.set_trace()
    S0_t, S0_p, S0_vars, thetas = estimate_probability_of_agreement(hhmm_time_low, hhmm_time_high, hhmm_censor_low, hhmm_censor_high, delta=delta)

    return S0_t, S0_p, S0_vars, thetas


def find_all_ps(cohort_range, ps=np.linspace(0, 1, num=11)):
    with open("model_res/cohort_range_{}_{}_baseline_vempirical.pickle".format(cohort_range[0], cohort_range[1]),
              "rb") as res:
        baseline_vempirical_df = pickle.load(res)

    # p_nonfrailty = compute_p_value(p_init, baseline_vempirical_df, option="nonfrailty")
    # p_frailty = compute_p_value(p_init, baseline_vempirical_df, option="frailty")
    # import pdb; pdb.set_trace()

    # def cons_f(p_threshold, baseline_vempirical_df, p_v=0.5):
    #     return compute_p_value(p_threshold, baseline_vempirical_df, option="nonfrailty")-p_v
    #
    # cons = ({'type': 'ineq', 'fun': cons_f, 'args': (baseline_vempirical_df, 0.5)})
    # prm_bounds = [(0,1)]
    # res = minimize(compute_p_value, p_init, method="SLSQP", constraints=cons, bounds=prm_bounds,
    #                args=(baseline_vempirical_df, "frailty"), options={'disp': True})

    p_fs = list()
    da_fs = list()
    p_ns = list()
    for p in ps:

        p_nonfrailty = compute_p_value(cohort_range, p, baseline_vempirical_df, option="nonfrailty")
        p_frailty, diff_average_frailty = compute_p_value(cohort_range, p, baseline_vempirical_df, option="frailty", diff_average=True)

        p_ns.append(p_nonfrailty)
        p_fs.append(p_frailty)
        da_fs.append(diff_average_frailty)

        # import pdb; pdb.set_trace()
    p_ns = np.array(p_ns)
    p_fs = np.array(p_fs)
    da_fs = np.array(da_fs)
    return p_ns, p_fs, da_fs


def find_all_prob_of_agreement_hhmm(cohort_range, ps=np.linspace(0, 1, num=11), delta=0.05):
    # S0_t, S0_p, theta = estimate_prob_of_agreement(cohort_range, 0.2, delta=delta)
    thetas_ps = list()
    S0_ts = list()
    for p in ps:
        S0_t, S0_p, S0_vars, thetas = estimate_prob_of_agreement_hhmm(cohort_range, p, delta=delta)
        thetas_ps.append(thetas)
        S0_ts.append(S0_t)
    return S0_ts, thetas_ps


def find_opt_p(ps, p_ns, p_fs, da_fs, alpha=0.05):
    opt_p = -1
    p_f_min = 1
    for p, p_n, p_f, da_f in zip(ps, p_ns, p_fs, da_fs):
        if (p_n > alpha and p_f < p_f_min) and da_f > 0:
            p_f_min = p_f
            opt_p = p
    return opt_p, p_ns[ps==opt_p], p_fs[ps==opt_p], da_fs[ps==opt_p]


def find_opt_p0(ps, p_ns, p_fs, da_fs, alpha=0.05):
    opt_p = -1
    p_n_min = 1
    for p, p_n, p_f, da_f in zip(ps, p_ns, p_fs, da_fs):
        if (p_f > alpha and p_n < p_n_min) and da_f > 0:
            p_n_min = p_n
            opt_p = p
    return opt_p, p_ns[ps == opt_p], p_fs[ps == opt_p], da_fs[ps == opt_p]


def find_opt_p_poa(ps, S0_t_list, theta_list):
    opt_theta = 1
    opt_p = 0
    opt_S0_t = None
    for p, S0_t, theta in zip(ps, S0_t_list, theta_list):
        if np.min(theta) < opt_theta:
            opt_theta = np.min(theta)
            opt_p = p
            opt_S0_t = S0_t
    return opt_p, opt_S0_t, opt_theta


def plot_KM(cohort_range, p_threshold, baseline_vempirical_df, dataset=None):
    model_df = extract_model_df(cohort_range, p_threshold=p_threshold, dataset=dataset)
    censor_threshold = None
    time, censoring, group = baseline_vempirical_df['time'].to_numpy(), baseline_vempirical_df['censoring'].to_numpy(), \
                             baseline_vempirical_df['group'].to_numpy()
    unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low = estimate_KM(
        time[group == 'low risk'], censoring[group == 'low risk'],
        censor_threshlod=censor_threshold)
    unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high = estimate_KM(
        time[group == 'high risk'], censoring[group == 'high risk'],
        censor_threshlod=censor_threshold)
    emp_time_low, emp_time_high, emp_censor_low, emp_censor_high = time[group == 'low risk'], time[
        group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']
    logrank_stats = logrank_test(emp_time_low, emp_time_high, emp_censor_low, emp_censor_high)
    print("baseline log p value = {}".format(np.round(np.log(logrank_stats.p_value), 2)))
    if censor_threshold is not None:
        censoring[time > censor_threshold] = 0
    # emp_time_low, emp_time_high, emp_censor_low, emp_censor_high = time[group == 'low risk'], time[
    #     group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']

    n_low_vempirical = (group == 'low risk').sum()
    n_high_vempirical = (group == 'high risk').sum()
    group_vempirical = group

    time, censoring, group = model_df['time'].to_numpy(), model_df['censoring'].to_numpy(), model_df['group'].to_numpy()
    unique_durations_model_low, survivals_model_low = estimate_KM(time[group == 'low risk'],
                                                                censoring[group == 'low risk'],
                                                                censor_threshlod=censor_threshold)
    unique_durations_model_high, survivals_model_high = estimate_KM(time[group == 'high risk'],
                                                                  censoring[group == 'high risk'],
                                                                  censor_threshlod=censor_threshold)
    hhmm_time_low, hhmm_time_high, hhmm_censor_low, hhmm_censor_high = time[group == 'low risk'], time[
        group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']
    logrank_stats = logrank_test(hhmm_time_low, hhmm_time_high, hhmm_censor_low, hhmm_censor_high)
    print("hhmm log p value = {}".format(np.round(np.log(logrank_stats.p_value), 2)))
    if censor_threshold is not None:
        censoring[time > censor_threshold] = 0
    # hhmm_time_low, hhmm_time_high, hhmm_censor_low, hhmm_censor_high = time[group == 'low risk'], time[
    #     group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']

    n_low_HHMM = (group == 'low risk').sum()
    n_high_HHMM = (group == 'high risk').sum()
    group_HHMM = group

    print("E(L) == M(L) {}; E(L) == M(H) {}".format(np.logical_and((group_vempirical == 'low risk'), (group_HHMM == 'low risk')).sum(),
                                                    np.logical_and((group_vempirical == 'low risk'), (group_HHMM == 'high risk')).sum()))
    print("E(H) == M(L) {}; E(H) == M(H) {}".format(np.logical_and((group_vempirical == 'high risk'), (group_HHMM == 'low risk')).sum(),
                                                    np.logical_and((group_vempirical == 'high risk'), (group_HHMM == 'high risk')).sum()))

    fig = plt.figure()
    plt.step(unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, where="post",
             label="empirical baseline: no frailty ({})".format(n_low_vempirical))
    plt.step(unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, where="post",
             label="empirical baseline: frailty ({})".format(n_high_vempirical))
    plt.step(unique_durations_model_low, survivals_model_low, where="post",
             label="HCTIHMM: no frailty ({})".format(n_low_HHMM))
    plt.step(unique_durations_model_high, survivals_model_high, where="post",
             label="HCTIHMM: frailty ({})".format(n_high_HHMM))
    plt.xlim(0, 20)
    plt.ylim(0.82, 1.)
    plt.legend(loc="lower left")
    plt.tight_layout()
    # plt.title("Kaplan Meier curves for baseline and HCTIHMM model")
    if censor_threshold is None:
        plt.savefig("model_res/{}/MC_CR{}_{}_threshold{}.png".format(dataset, cohort_range[0], cohort_range[1], p_threshold))
    else:
        plt.xlim(0, censor_threshold)
        plt.savefig(
            "model_res/{}/MC_CR{}_{}_threshold{}_c{}.png".format(dataset, cohort_range[0], cohort_range[1], p_threshold, censor_threshold))
    plt.show()
    plt.close(fig)

    # save figure data
    with open("model_res/{}/MC_CR{}_{}_threshold{}.pkl".format(dataset, cohort_range[0], cohort_range[1], p_threshold), "wb") as res:
        pickle.dump([unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, unique_durations_baseline_vempirical_high,
                     survivals_baseline_vempirical_high, unique_durations_model_low, survivals_model_low, unique_durations_model_high, survivals_model_high], res)


def plot_probability_of_agreement(cohort_range, p_threshold, baseline_vempirical_df, dataset=None, delta=0.05):
    model_df = extract_model_df(cohort_range, p_threshold=p_threshold, dataset=dataset)
    censor_threshold = None

    time, censoring, group = baseline_vempirical_df['time'].to_numpy(), baseline_vempirical_df['censoring'].to_numpy(), \
                             baseline_vempirical_df['group'].to_numpy()
    if censor_threshold is not None:
        censoring[time > censor_threshold] = 0
    emp_time_low, emp_time_high, emp_censor_low, emp_censor_high = time[group == 'low risk'], time[
        group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']
    emp_S0_t, emp_S0_p, emp_S0_vars, emp_thetas = estimate_probability_of_agreement(emp_time_low, emp_time_high, emp_censor_low,
                                                                    emp_censor_high, delta=delta)

    time, censoring, group = model_df['time'].to_numpy(), model_df['censoring'].to_numpy(), model_df['group'].to_numpy()
    if censor_threshold is not None:
        censoring[time > censor_threshold] = 0
    hhmm_time_low, hhmm_time_high, hhmm_censor_low, hhmm_censor_high = time[group == 'low risk'], time[
        group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']
    hhmm_S0_t, hhmm_S0_p, hhmm_S0_vars, hhmm_thetas = estimate_probability_of_agreement(hhmm_time_low, hhmm_time_high, hhmm_censor_low,
                                                                    hhmm_censor_high, delta=delta)

    # import pdb; pdb.set_trace()
    fig = plt.figure()
    plt.step(emp_S0_t, emp_thetas, where="post",
             label="Empirical baseline")
    plt.step(hhmm_S0_t, hhmm_thetas, where="post",
             label="HCTIHMM")
    plt.xlim(0, 20)
    # plt.ylim(0., 1.)
    plt.legend(loc="lower left")
    plt.tight_layout()
    # plt.title("Kaplan Meier curves for baseline and HCTIHMM model")
    if censor_threshold is None:
        plt.savefig(
            "model_res/{}/prob_of_agreement_{}_{}_threshold{}.png".format(dataset, cohort_range[0], cohort_range[1], p_threshold))
    else:
        plt.xlim(0, censor_threshold)
        plt.savefig(
            "model_res/{}/prob_of_agreement_{}_{}_threshold{}.png".format(dataset, cohort_range[0], cohort_range[1], p_threshold,
                                                                 censor_threshold))
    plt.show()
    plt.close(fig)

    # save figure data
    with open("model_res/{}/prob_of_agreement_{}_{}_threshold{}.pkl".format(dataset, cohort_range[0], cohort_range[1], p_threshold), "wb") as res:
        pickle.dump([emp_S0_t, emp_thetas, hhmm_S0_t, hhmm_thetas], res)

    # import pdb; pdb.set_trace()


if __name__ == "__main__":

    train_dataset = "data_random_240k"
    test_dataset = "data_random_240k_test"

    cohort_ranges = [[30, 34], [35, 39], [40, 44], [45, 49], [50, 54], [55, 59], [60, 64], [65, 69]]

    # ################################################################
    # ### optimization framework based on probability of agreement ###
    # ################################################################
    #
    # ps = np.linspace(0.1, 0.9, num=81)
    # # S0_ts = list()
    # # thetas = list()
    # # for index_range in range(8):
    # #     print("{}th cohort.".format(index_range))
    # #     s, theta = find_all_prob_of_agreement_hhmm(cohort_range=cohort_ranges[index_range], ps=ps, delta=0.1)
    # #     S0_ts.append(s)
    # #     thetas.append(theta)
    # # with open("res/ps100_prob_of_agreement.pickle", "wb") as f:
    # #     pickle.dump([S0_ts, thetas], f)
    # with open("res/ps100_prob_of_agreement.pickle", "rb") as f:
    #     S0_ts, thetas = pickle.load(f)
    #
    # for index_range in range(8):
    #     cohort_range = cohort_ranges[index_range]
    #     opt_p, opt_S0_t, opt_theta = find_opt_p_poa(ps, S0_ts[index_range], thetas[index_range])
    #     print(opt_p)
    # import pdb; pdb.set_trace()

    #######################################
    ### original optimization framework ###
    #######################################

    # p_nonfrailtys = list()
    # p_frailtys = list()
    # da_frailtys = list()
    # ps = np.linspace(0, 1, num=101)
    # for index_range in range(8):
    #     print("{}th cohort.".format(index_range))
    #     p_ns, p_fs, da_fs = find_all_ps(cohort_range=cohort_ranges[index_range], ps=ps)
    #     p_nonfrailtys.append(p_ns)
    #     p_frailtys.append(p_fs)
    #     da_frailtys.append(da_fs)
    # p_nonfrailtys = np.stack(p_nonfrailtys)
    # p_frailtys = np.stack(p_frailtys)
    # da_frailtys = np.stack(da_frailtys)
    # with open("res/ps100.pickle", "wb") as f:
    #     pickle.dump([ps, p_nonfrailtys, p_frailtys, da_frailtys], f)
    with open("res/ps100.pickle", "rb") as res:
        ps, p_nonfrailtys, p_frailtys, da_frailtys = pickle.load(res)

    option = 1
    # option = 2

    for index_range in range(8):
        cohort_range = cohort_ranges[index_range]
        if option == 1:
            opt_p, opt_p_n, opt_p_f, opt_da_f = find_opt_p(ps, p_nonfrailtys[index_range], p_frailtys[index_range],
                                                           da_frailtys[index_range], alpha=0.1)
            print(cohort_ranges[index_range])
            print(opt_p, opt_p_n, opt_p_f, opt_da_f)
        if option == 2:
            opt_p, opt_p_n, opt_p_f, opt_da_f = find_opt_p0(ps, p_nonfrailtys[index_range], p_frailtys[index_range],
                                                           da_frailtys[index_range], alpha=0.1)
            print(cohort_ranges[index_range])
            print(opt_p, opt_p_n, opt_p_f, opt_da_f)
        with open("model_res/{}/cohort_range_{}_{}_baseline_vempirical.pickle".format(train_dataset, cohort_range[0], cohort_range[1]),"rb") as res:
                baseline_vempirical_df = pickle.load(res)
        plot_KM(cohort_ranges[index_range], opt_p, baseline_vempirical_df, dataset=train_dataset)
        with open("model_res/{}/cohort_range_{}_{}_baseline_vempirical.pickle".format(test_dataset, cohort_range[0], cohort_range[1]),"rb") as res:
                baseline_vempirical_df_test = pickle.load(res)
        plot_KM(cohort_ranges[index_range], opt_p, baseline_vempirical_df_test, dataset=test_dataset)
        plot_probability_of_agreement(cohort_ranges[index_range], opt_p, baseline_vempirical_df_test, dataset=test_dataset, delta=0.05)
    import pdb; pdb.set_trace()

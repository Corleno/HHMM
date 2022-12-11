import pickle
import matplotlib.pyplot as plt
import numpy as np
from lifelines.statistics import logrank_test
from hhmm_application import *
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS



def Create_KM(durations, event_observed, censor_threshlod=None, verbose=False):
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
    return unique_durations, survivals


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
    unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low = Create_KM(
        time[group == 'low risk'], censoring[group == 'low risk'],
        censor_threshlod=censor_threshold)
    unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high = Create_KM(
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
    unique_durations_model_low, survivals_model_low = Create_KM(time[group == 'low risk'],
                                                                censoring[group == 'low risk'],
                                                                censor_threshlod=censor_threshold)
    unique_durations_model_high, survivals_model_high = Create_KM(time[group == 'high risk'],
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


def compute_p_value(cohort_range, p_threshold, baseline_vempirical_df=None, option="nonfrailty", diff_average=False):
    model_df = extract_model_df(cohort_range, p_threshold=p_threshold)
    censor_threshold = None

    if baseline_vempirical_df is not None:
        time, censoring, group = baseline_vempirical_df['time'].to_numpy(), baseline_vempirical_df['censoring'].to_numpy(), \
                                 baseline_vempirical_df['group'].to_numpy()
        # unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low = Create_KM(
        #     time[group == 'low risk'], censoring[group == 'low risk'],
        #     censor_threshlod=censor_threshold)
        # unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high = Create_KM(
        #     time[group == 'high risk'], censoring[group == 'high risk'],
        #     censor_threshlod=censor_threshold)
        if censor_threshold is not None:
            censoring[time > censor_threshold] = 0
        emp_time_low, emp_time_high, emp_censor_low, emp_censor_high = time[group == 'low risk'], time[
            group == 'high risk'], censoring[group == 'low risk'], censoring[group == 'high risk']

    time, censoring, group = model_df['time'].to_numpy(), model_df['censoring'].to_numpy(), model_df['group'].to_numpy()
    # unique_durations_model_low, survivals_model_low = Create_KM(time[group == 'low risk'],
    #                                                             censoring[group == 'low risk'],
    #                                                             censor_threshlod=censor_threshold)
    # unique_durations_model_high, survivals_model_high = Create_KM(time[group == 'high risk'],
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
    if option == "hhmm":
        results = logrank_test(hhmm_time_low, hhmm_time_high, hhmm_censor_low, hhmm_censor_high)
        print('hhmm: ', np.round(results.p_value, 2))
        if diff_average:
            return results.p_value, hhmm_time_low[hhmm_censor_low==1].mean() - hhmm_time_high[
                hhmm_censor_high==1].mean()
        else:
            return results.p_value


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

    p_hhmms = list()
    d_hhmms = list()
    for p in ps:
        p_hhmm, d_hhmm = compute_p_value(cohort_range, p, option="hhmm", diff_average=True)
        p_hhmms.append(p_hhmm)
        d_hhmms.append(d_hhmm)
        # import pdb; pdb.set_trace()
    p_hhmms = np.array(p_hhmms)
    d_hhmms = np.array(d_hhmms)
    return p_hhmms, d_hhmms


def find_opt_p(ps, p_hhmms, d_hhmms):

    opt_p = -1
    p_hhmm_min = 1
    for p, p_hhmm, d_hhmm in zip(ps, p_hhmms, d_hhmms):
        # import pdb; pdb.set_trace()
        if p_hhmm < p_hhmm_min: # and d_hhmm > 0:
            p_hhmm_min = p_hhmm
            opt_p = p

    return opt_p, p_hhmms[ps==opt_p], d_hhmms[ps==opt_p]


def plot_KM(cohort_range, p_threshold, baseline_vempirical_df, dataset, do_plot=True):
    model_df = extract_model_df(cohort_range, p_threshold=p_threshold, dataset=dataset)
    censor_threshold = None
    time, censoring, group = baseline_vempirical_df['time'].to_numpy(), baseline_vempirical_df['censoring'].to_numpy(), \
                             baseline_vempirical_df['group'].to_numpy()
    unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low = Create_KM(
        time[group == 'low risk'], censoring[group == 'low risk'],
        censor_threshlod=censor_threshold)
    unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high = Create_KM(
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

    time, censoring, group = model_df['time'].to_numpy(), model_df['censoring'].to_numpy(), model_df['group'].to_numpy()
    unique_durations_model_low, survivals_model_low = Create_KM(time[group == 'low risk'],
                                                                censoring[group == 'low risk'],
                                                                censor_threshlod=censor_threshold)
    unique_durations_model_high, survivals_model_high = Create_KM(time[group == 'high risk'],
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

    # nonfrailty test
    logrank_stats = logrank_test(emp_time_low, hhmm_time_low, emp_censor_low, hhmm_censor_low)
    print("nonfrailty p value = {}".format(np.round((logrank_stats.p_value), 4)))
    # frailty test
    logrank_stats = logrank_test(emp_time_high, hhmm_time_high, emp_censor_high, hhmm_censor_high)
    print("frailty p value = {}".format(np.round((logrank_stats.p_value), 4)))

    if do_plot:
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


if __name__ == "__main__":
    train_dataset = "data_random_240k"
    test_dataset = "data_random_240k_test"

    cohort_ranges = [[30, 34], [35, 39], [40, 44], [45, 49], [50, 54], [55, 59], [60, 64], [65, 69]]
    # p_hhmms = list()
    # d_hhmms = list()
    # ps = np.linspace(0, 1, num=101)
    # for index_range in range(8):
    #     print("{}th cohort.".format(index_range))
    #     p_hhmm, d_hhmm = find_all_ps(cohort_range=cohort_ranges[index_range], ps=ps)
    #     p_hhmms.append(p_hhmm)
    #     d_hhmms.append(d_hhmm)
    # p_hhmms = np.stack(p_hhmms)
    # d_hhmms = np.stack(d_hhmms)
    # with open("res/ps_hhmm_100.pickle", "wb") as f:
    #     pickle.dump([ps, p_hhmms, d_hhmms], f)
    with open("res/ps_hhmm_100.pickle", "rb") as res:
        ps, p_hhmms, d_hhmms = pickle.load(res)
    with open("res/ps100.pickle", "rb") as res:
        ps, p_nonfrailtys, p_frailtys, da_frailtys = pickle.load(res)

    for index_range in range(8):
        cohort_range = cohort_ranges[index_range]
        p_ns = p_nonfrailtys[index_range]
        p_fs = p_frailtys[index_range]
        opt_p, opt_p_hhmm, opt_d_hhmm = find_opt_p(ps, p_hhmms[index_range], d_hhmms[index_range])
        print(cohort_ranges[index_range])
        print(opt_p, np.log(opt_p_hhmm), opt_d_hhmm, p_ns[ps == opt_p], p_fs[ps == opt_p])
        with open("model_res/{}/cohort_range_{}_{}_baseline_vempirical.pickle".format(train_dataset, cohort_range[0], cohort_range[1]),
                  "rb") as res:
            baseline_vempirical_df = pickle.load(res)
        plot_KM(cohort_ranges[index_range], opt_p, baseline_vempirical_df, train_dataset, do_plot=False)
        with open("model_res/{}/cohort_range_{}_{}_baseline_vempirical.pickle".format(test_dataset, cohort_range[0], cohort_range[1]),"rb") as res:
                baseline_vempirical_df_test = pickle.load(res)
        plot_KM(cohort_ranges[index_range], opt_p, baseline_vempirical_df_test, dataset=test_dataset, do_plot=False)


    import pdb; pdb.set_trace()

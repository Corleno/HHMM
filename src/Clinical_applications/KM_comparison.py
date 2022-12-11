import pickle
import matplotlib.pyplot as plt
import numpy as np
from lifelines.statistics import logrank_test


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

    return


if __name__ == "__main__":
    p_threshold = 0.4
    for index_range in range(8):
        main(index_range, p_threshold)

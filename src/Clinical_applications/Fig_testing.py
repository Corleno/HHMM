import matplotlib.pyplot as plt
import pickle
import numpy as np


if __name__ == "__main__":

    title_size = 15
    tick_size = 15
    label_size = 15
    legend_size = 15
    fig, axs = plt.subplots(4, 4, figsize=(15, 16))

    ax = axs[0, 0]
    with open("model_res/data_random_240k_test/MC_CR30_34_threshold0.33.pkl", "rb") as res:
        unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, unique_durations_model_low, survivals_model_low, unique_durations_model_high, survivals_model_high = pickle.load(res)
    ax.step(unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, where="post",
             label="EM(N)")
    ax.step(unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, where="post",
             label="EM(F)")
    ax.step(unique_durations_model_low, survivals_model_low, where="post",
             label="HCTIHMM(N)")
    ax.step(unique_durations_model_high, survivals_model_high, where="post",
             label="HCTIHMM(F)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0.8, 1.)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.8, 1, num=5))
    ax.set_yticklabels(np.round(np.linspace(0.8, 1, num=5),2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    ax.set_ylabel("Event probability", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)
    ax.set_title("Cohort (30-34)", fontsize=title_size)

    ax = axs[1, 0]
    with open("model_res/data_random_240k_test/prob_of_agreement_30_34_threshold0.33.pkl", "rb") as res:
        emp_S0_t, emp_thetas, hhmm_S0_t, hhmm_thetas = pickle.load(res)
    ax.step(emp_S0_t, emp_thetas, where="post",
            label="EM(PoA)", linestyle='--', linewidth=3.0)
    ax.step(hhmm_S0_t, hhmm_thetas, where="post",
            label="HCTIHMM(PoA)", linestyle='--', linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0, 1, num=6))
    ax.set_yticklabels(np.round(np.linspace(0, 1, num=6), 1), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    ax.set_ylabel("Probability of agreement", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)

    ax = axs[0, 1]
    with open("model_res/data_random_240k_test/MC_CR35_39_threshold0.33.pkl", "rb") as res:
        unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, unique_durations_model_low, survivals_model_low, unique_durations_model_high, survivals_model_high = pickle.load(res)
    ax.step(unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, where="post",
            label="EM(N)")
    ax.step(unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, where="post",
            label="EM(F)")
    ax.step(unique_durations_model_low, survivals_model_low, where="post",
            label="HCTIHMM(N)")
    ax.step(unique_durations_model_high, survivals_model_high, where="post",
            label="HCTIHMM(F)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0.8, 1.)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.8, 1, num=5))
    ax.set_yticklabels(np.round(np.linspace(0.8, 1, num=5), 2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Event probability", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)
    ax.set_title("Cohort (35-39)", fontsize=title_size)

    ax = axs[1, 1]
    with open("model_res/data_random_240k_test/prob_of_agreement_35_39_threshold0.33.pkl", "rb") as res:
        emp_S0_t, emp_thetas, hhmm_S0_t, hhmm_thetas = pickle.load(res)
    ax.step(emp_S0_t, emp_thetas, where="post",
            label="EM(PoA)", linestyle='--', linewidth=3.0)
    ax.step(hhmm_S0_t, hhmm_thetas, where="post",
            label="HCTIHMM(PoA)", linestyle='--', linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Probability of agreement", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)

    ax = axs[0, 2]
    with open("model_res/data_random_240k_test/MC_CR40_44_threshold0.42.pkl", "rb") as res:
        unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, unique_durations_model_low, survivals_model_low, unique_durations_model_high, survivals_model_high = pickle.load(
            res)
    ax.step(unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, where="post",
            label="EM(N)")
    ax.step(unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, where="post",
            label="EM(F)")
    ax.step(unique_durations_model_low, survivals_model_low, where="post",
            label="HCTIHMM(N)")
    ax.step(unique_durations_model_high, survivals_model_high, where="post",
            label="HCTIHMM(F)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0.8, 1.)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.8, 1, num=5))
    ax.set_yticklabels(np.round(np.linspace(0.8, 1, num=5), 2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Event probability", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)
    ax.set_title("Cohort (40-44)", fontsize=title_size)

    ax = axs[1, 2]
    with open("model_res/data_random_240k_test/prob_of_agreement_40_44_threshold0.42.pkl", "rb") as res:
        emp_S0_t, emp_thetas, hhmm_S0_t, hhmm_thetas = pickle.load(res)
    ax.step(emp_S0_t, emp_thetas, where="post",
            label="EM(PoA)", linestyle='--', linewidth=3.0)
    ax.step(hhmm_S0_t, hhmm_thetas, where="post",
            label="HCTIHMM(PoA)", linestyle='--', linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Probability of agreement", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)

    ax = axs[0, 3]
    with open("model_res/data_random_240k_test/MC_CR45_49_threshold0.44.pkl", "rb") as res:
        unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, unique_durations_model_low, survivals_model_low, unique_durations_model_high, survivals_model_high = pickle.load(
            res)
    ax.step(unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, where="post",
            label="EM(N)")
    ax.step(unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, where="post",
            label="EM(F)")
    ax.step(unique_durations_model_low, survivals_model_low, where="post",
            label="HCTIHMM(N)")
    ax.step(unique_durations_model_high, survivals_model_high, where="post",
            label="HCTIHMM(F)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0.8, 1.)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.8, 1, num=5))
    ax.set_yticklabels(np.round(np.linspace(0.8, 1, num=5), 2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Event probability", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)
    ax.set_title("Cohort (45-49)", fontsize=title_size)

    ax = axs[1, 3]
    with open("model_res/data_random_240k_test/prob_of_agreement_45_49_threshold0.44.pkl", "rb") as res:
        emp_S0_t, emp_thetas, hhmm_S0_t, hhmm_thetas = pickle.load(res)
    ax.step(emp_S0_t, emp_thetas, where="post",
            label="EM(PoA)", linestyle='--', linewidth=3.0)
    ax.step(hhmm_S0_t, hhmm_thetas, where="post",
            label="HCTIHMM(PoA)", linestyle='--', linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Probability of agreement", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)

    ax = axs[2, 0]
    with open("model_res/data_random_240k_test/MC_CR50_54_threshold0.53.pkl", "rb") as res:
        unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, unique_durations_model_low, survivals_model_low, unique_durations_model_high, survivals_model_high = pickle.load(res)
    ax.step(unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, where="post",
             label="EM(N)")
    ax.step(unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, where="post",
             label="EM(F)")
    ax.step(unique_durations_model_low, survivals_model_low, where="post",
             label="HCTIHMM(N)")
    ax.step(unique_durations_model_high, survivals_model_high, where="post",
             label="HCTIHMM(F)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0.8, 1.)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.8, 1, num=5))
    ax.set_yticklabels(np.round(np.linspace(0.8, 1, num=5),2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    ax.set_ylabel("Event probability", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)
    ax.set_title("Cohort (50-54)", fontsize=title_size)

    ax = axs[3, 0]
    with open("model_res/data_random_240k_test/prob_of_agreement_50_54_threshold0.53.pkl", "rb") as res:
        emp_S0_t, emp_thetas, hhmm_S0_t, hhmm_thetas = pickle.load(res)
    ax.step(emp_S0_t, emp_thetas, where="post",
            label="EM(PoA)", linestyle='--', linewidth=3.0)
    ax.step(hhmm_S0_t, hhmm_thetas, where="post",
            label="HCTIHMM(PoA)", linestyle='--', linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.75, 1, num=6))
    ax.set_yticklabels(np.round(np.linspace(0.75, 1, num=6),2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    ax.set_ylabel("Probability of agreement", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)

    ax = axs[2, 1]
    with open("model_res/data_random_240k_test/MC_CR55_59_threshold0.51.pkl", "rb") as res:
        unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, unique_durations_model_low, survivals_model_low, unique_durations_model_high, survivals_model_high = pickle.load(res)
    ax.step(unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, where="post",
            label="EM(N)")
    ax.step(unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, where="post",
            label="EM(F)")
    ax.step(unique_durations_model_low, survivals_model_low, where="post",
            label="HCTIHMM(N)")
    ax.step(unique_durations_model_high, survivals_model_high, where="post",
            label="HCTIHMM(F)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0.8, 1.)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.8, 1, num=5))
    ax.set_yticklabels(np.round(np.linspace(0.8, 1, num=5), 2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Event probability", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)
    ax.set_title("Cohort (55-59)", fontsize=title_size)

    ax = axs[3, 1]
    with open("model_res/data_random_240k_test/prob_of_agreement_55_59_threshold0.51.pkl", "rb") as res:
        emp_S0_t, emp_thetas, hhmm_S0_t, hhmm_thetas = pickle.load(res)
    ax.step(emp_S0_t, emp_thetas, where="post",
            label="EM(PoA)", linestyle='--', linewidth=3.0)
    ax.step(hhmm_S0_t, hhmm_thetas, where="post",
            label="HCTIHMM(PoA)", linestyle='--', linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.98, 1, num=3))
    ax.set_yticklabels(np.round(np.linspace(0.98, 1, num=3), 2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Probability of agreement", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)

    ax = axs[2, 2]
    with open("model_res/data_random_240k_test/MC_CR60_64_threshold0.82.pkl", "rb") as res:
        unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, unique_durations_model_low, survivals_model_low, unique_durations_model_high, survivals_model_high = pickle.load(
            res)
    ax.step(unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, where="post",
            label="EM(N)")
    ax.step(unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, where="post",
            label="EM(F)")
    ax.step(unique_durations_model_low, survivals_model_low, where="post",
            label="HCTIHMM(N)")
    ax.step(unique_durations_model_high, survivals_model_high, where="post",
            label="HCTIHMM(F)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0.8, 1.)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.8, 1, num=5))
    ax.set_yticklabels(np.round(np.linspace(0.8, 1, num=5), 2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Event probability", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)
    ax.set_title("Cohort (60-64)", fontsize=title_size)

    ax = axs[3, 2]
    with open("model_res/data_random_240k_test/prob_of_agreement_60_64_threshold0.82.pkl", "rb") as res:
        emp_S0_t, emp_thetas, hhmm_S0_t, hhmm_thetas = pickle.load(res)
    ax.step(emp_S0_t, emp_thetas, where="post",
            label="EM(PoA)", linestyle='--', linewidth=3.0)
    ax.step(hhmm_S0_t, hhmm_thetas, where="post",
            label="HCTIHMM(PoA)", linestyle='--', linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    ax.set_yticks(np.linspace(0.99, 1, num=2))
    ax.set_yticklabels(np.round(np.linspace(0.99, 1, num=2), 2), fontsize=tick_size)
    # ax.set_ylabel("Probability of agreement", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)

    ax = axs[2, 3]
    with open("model_res/data_random_240k_test/MC_CR65_69_threshold0.94.pkl", "rb") as res:
        unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, unique_durations_model_low, survivals_model_low, unique_durations_model_high, survivals_model_high = pickle.load(
            res)
    ax.step(unique_durations_baseline_vempirical_low, survivals_baseline_vempirical_low, where="post",
            label="EA(N)")
    ax.step(unique_durations_baseline_vempirical_high, survivals_baseline_vempirical_high, where="post",
            label="EA(F)")
    ax.step(unique_durations_model_low, survivals_model_low, where="post",
            label="HCTIHMM(N)")
    ax.step(unique_durations_model_high, survivals_model_high, where="post",
            label="HCTIHMM(F)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0.8, 1.)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.8, 1, num=5))
    ax.set_yticklabels(np.round(np.linspace(0.8, 1, num=5), 2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Event probability", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)
    ax.set_title("Cohort (65-69)", fontsize=title_size)

    ax = axs[3, 3]
    with open("model_res/data_random_240k_test/prob_of_agreement_65_69_threshold0.94.pkl", "rb") as res:
        emp_S0_t, emp_thetas, hhmm_S0_t, hhmm_thetas = pickle.load(res)
    ax.step(emp_S0_t, emp_thetas, where="post",
            label="EA(PoA)", linestyle='--', linewidth=3.0)
    ax.step(hhmm_S0_t, hhmm_thetas, where="post",
            label="HCTIHMM(PoA)", linestyle='--', linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.linspace(0, 20, num=5))
    ax.set_xticklabels(np.linspace(0, 20, num=5).astype(int), fontsize=tick_size)
    ax.set_yticks(np.linspace(0.75, 1, num=6))
    ax.set_yticklabels(np.round(np.linspace(0.75, 1, num=6), 2), fontsize=tick_size)
    ax.set_xlabel("Year", fontsize=label_size)
    # ax.set_ylabel("Probability of agreement", fontsize=label_size)
    # ax.legend(loc="lower left", fontsize=legend_size)

    # Put a legend below current axis
    lines_labels_s = axs[2, 3].get_legend_handles_labels()
    lines_labels_poa = axs[3, 3].get_legend_handles_labels()

    lines, labels = lines_labels_s[0] + lines_labels_poa[0], lines_labels_s[1] + lines_labels_poa[1]
    fig.subplots_adjust(bottom=0.1, hspace=0.35, wspace=0.3)
    # fig.legend(lines, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=legend_size)
    fig.legend(lines, labels, loc='lower center', ncol=6, fontsize=legend_size)

    # plt.tight_layout()
    # plt.savefig("fig_testing.png")
    plt.savefig("fig_testing.eps", format='eps')
    plt.show()

    # import pdb; pdb.set_trace()


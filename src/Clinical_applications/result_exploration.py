import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Alphas = []
    Etas = []
    Ws = []
    Cs = []
    nTests = 3
    nStates = 4

    with open("../../res/EM_240000/EM_hierarchical_16_updated_data_inv4_continuous_240000/res", "rb") as f:
        counter, currZ_pos_list, currStates_list, currAlpha, currEta, currW, currC, currNegLogLik, inv = pickle.load(f)

    currZ_pos = np.concatenate(currZ_pos_list)
    f, axis = plt.subplots(2, 1, sharex=True)
    ax = f.add_subplot(111, frameon=False)
    ax1 = axis[0]
    ax2 = axis[1]
    ax.plot(range(1))
    ax.set_xlabel("Posterior probability", fontsize=20, labelpad=20)
    ax.set_ylabel("Density", fontsize=20, labelpad=30)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax1.hist(currZ_pos, density=True, bins=100)
    ax1.set_ylim(1, 66)
    ax1.tick_params(labelsize=16)
    ax2.hist(currZ_pos, density=True, bins=100)
    ax2.set_ylim(0, 1)
    ax2.tick_params(labelsize=16)
    # plt.savefig("posterior_dist_frailty_v1.png", dpi=500)
    plt.savefig("posterior_dist_frailty_v1.eps", dpi=500, format='eps')
    plt.show()
    # import pdb
    # pdb.set_trace()

    # fig = plt.figure()
    # plt.hist(currZ_pos, density=True, bins=100)
    # plt.xlabel("Posterior probability", fontsize=18)
    # plt.ylabel("Density", fontsize=18)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.tight_layout()
    # plt.savefig("posterior_dist_frailty.png")


    # for seed in range(10):
    #     with open("../res/EM_hierarchical_16_updated_data_inv4_continuous_240000/res_p_0.2seed_{}".format(seed), "rb") as f:
    #         counter, currZ_pos_list, currStates_list, currAlpha, currEta, currW, currC, currNegLogLik, inv = pickle.load(f)
    #     Alpha = currAlpha[1]
    #     for k in range(nTests):
    #         for s in range(nStates):
    #             # import pdb
    #             # pdb.set_trace()
    #             Alpha[k][s, :] = np.exp(Alpha[k][s, :])
    #             Alpha[k][s, :] /= np.sum(Alpha[k][s, :])
    #     Eta = np.exp(currEta[1])
    #     W = [np.exp(currW[0]), np.exp(currW[1])]
    #     C = [np.exp(currC[0]), np.exp(currC[1])]
    #     C = [C[0]/np.sum(C[0], axis=0), C[1]/np.sum(C[1], axis=0)]
    #     Alphas.append(Alpha)
    #     Etas.append(Eta)
    #     Ws.append(W)
    #     Cs.append(C)
    # import pdb
    # pdb.set_trace()
    # Alpha0s = np.stack([Alphas[seed][0] for seed in range(10)])
    # Alpha0_mean, Alpha0_std = np.mean(Alpha0s, axis=0), np.std(Alpha0s, axis=0)
    # Alpha1s = np.stack([Alphas[seed][1] for seed in range(10)])
    # Alpha1_mean, Alpha1_std = np.mean(Alpha1s, axis=0), np.std(Alpha1s, axis=0)
    # Alpha2s = np.stack([Alphas[seed][2] for seed in range(10)])
    # Alpha2_mean, Alpha2_std = np.mean(Alpha2s, axis=0), np.std(Alpha2s, axis=0)
    # print("Diagnositic test result probabilities conditioned on hidden state")
    # print(np.round(Alpha0_mean, 4))
    # print(np.round(Alpha0_std, 4))
    # print(np.round(Alpha1_mean, 4))
    # print(np.round(Alpha1_std, 4))
    # print(np.round(Alpha2_mean, 4))
    # print(np.round(Alpha2_std, 4))
    # Etas = np.stack(Etas)
    # Eta_mean, Eta_std = np.mean(Etas, axis=0), np.std(Etas, axis=0)
    # print("Poisson intensities for the number of tests")
    # print(np.round(Eta_mean, 4))
    # print(np.round(Eta_std, 4))
    # W0s = np.stack([Ws[seed][0] for seed in range(10)])
    # W0_mean, W0_std = np.mean(W0s, axis=0), np.std(W0s, axis=0)
    # W1s = np.stack([Ws[seed][1] for seed in range(10)])
    # W1_mean, W1_std = np.mean(W1s, axis=0), np.std(W1s, axis=0)
    # print("Age dependent transition intensities")
    # print(np.round(W0_mean,4))
    # print(np.round(W0_std,4))
    # print(np.round(W1_mean,4))
    # print(np.round(W1_std,4))
    # print("Probabilities of being a particular state at the first screening")
    # C0s = np.stack([Cs[seed][0] for seed in range(10)])
    # C0_mean, C0_std = np.mean(C0s, axis=0), np.std(C0s, axis=0)
    # C1s = np.stack([Cs[seed][1] for seed in range(10)])
    # C1_mean, C1_std = np.mean(C1s, axis=0), np.std(C1s, axis=0)
    # print(np.round(C0_mean, 4))
    # print(np.round(C0_std, 4))
    # print(np.round(C1_mean, 4))
    # print(np.round(C1_std, 4))
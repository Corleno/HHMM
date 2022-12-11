import numpy as np
from scipy.linalg import expm
import pickle

if __name__ == "__main__":
    N_1 = 200
    N_2 = 300
    # N_1 = 500
    # N_2 = 500
    N = 50

    F_1 = np.array([[-0.1, 0.1], [0.1, -0.1]])
    # F_2 = np.array([[-0.05, 0.05], [0.05, -0.05]])
    F_2 = np.array([[-1, 1], [1, -1]])
    emission0 = np.array([0.05, 0.95])
    emission1 = np.array([0.95, 0.05])
    np.random.seed(22)

    # simulate N_1 time series with length N
    print("Type 1")
    Ss1 = []
    t1 = []
    obs1 = []
    for i in range(N_1):
        # simulate N-1 time stamps
        ts = np.sort(np.random.uniform(0, 10, N-1))
        ts = np.concatenate([[0], ts])
        # print(ts)
        Ss = [0]
        obs = [np.random.choice(2, p=emission0)]
        for j in range(N-1):
            if ts[j+1] < 5:
                delta = ts[j+1] - ts[j]
                T = expm(F_1*delta)
            if ts[j] < 5 and ts[j+1] > 5:
                T = np.matmul(expm(F_1*(5 - ts[j])), expm(F_2*(ts[j+1] - 5)))
            if ts[j] > 5:
                delta = ts[j+1] - ts[j]
                T = expm(F_2*delta)
            Ss.append(np.random.choice(2, p = T[Ss[-1]]))
            if Ss[-1] == 0:
                obs.append(np.random.choice(2, p=emission0))
            else:
                obs.append(np.random.choice(2, p=emission1))
        print("ts", ts, "Ss", Ss)
        Ss1.append(Ss)
        t1.append(ts)
        obs1.append(obs)
        # import pdb
        # pdb.set_trace()

    # simulate N_2 time series with length N
    print("Type 2")
    Ss2 = []
    t2 = []
    obs2 = []
    for i in range(N_2):
        # simulate N-1 time stamps
        ts = np.sort(np.random.uniform(0, 10, N-1))
        ts = np.concatenate([[0], ts])
        # print(ts)
        Ss = [0]
        obs = [np.random.choice(2, p=emission0)]
        for j in range(N-1):
            if ts[j+1] < 5:
                delta = ts[j+1] - ts[j]
                T = expm(F_2*delta)
            if ts[j] < 5 and ts[j+1] > 5:
                T = np.matmul(expm(F_2*(5 - ts[j])), expm(F_1*(ts[j+1] - 5)))
            if ts[j] > 5:
                delta = ts[j+1] - ts[j]
                T = expm(F_1*delta)
            Ss.append(np.random.choice(2, p = T[Ss[-1]]))
            if Ss[-1] == 0:
                obs.append(np.random.choice(2, p=emission0))
            else:
                obs.append(np.random.choice(2, p=emission1))
        print("ts", ts, "Ss", Ss)
        Ss2.append(Ss)
        t2.append(ts)
        obs2.append(obs)
    print(Ss2)

    Ss1 = np.array(Ss1)
    Ss2 = np.array(Ss2)
    obs1 = np.array(obs1)
    obs2 = np.array(obs2)
    t1 = np.stack(t1)
    t2 = np.stack(t2)

    # with open("synthetic_data2.pickle", "wb") as res: # 200 vs 300
    #     pickle.dump([np.concatenate([t1, t2], axis=0), np.concatenate([obs1, obs2], axis=0), np.concatenate([Ss1, Ss2], axis=0)], res)

    with open("synthetic_data_test.pickle", "wb") as res: # 500 vs 500
        pickle.dump([np.concatenate([t1, t2], axis=0), np.concatenate([obs1, obs2], axis=0), np.concatenate([Ss1, Ss2], axis=0)], res)

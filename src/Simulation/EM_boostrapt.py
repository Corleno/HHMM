import numpy as np 
from scipy.linalg import expm
import pickle
from scipy.optimize import minimize

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def prob_obs(p, obs):
    return p*obs+(1-p)*(1-obs)

def FFBS(alpha, beta, t, obs):
    """
    Forward Backward algorithm to compute the log likelihood of observations
    """
    # create transition intensity matrix
    Q0 = np.array([[-np.exp(alpha[0]), np.exp(alpha[0])], [np.exp(alpha[1]), -np.exp(alpha[1])]])
    Q1 = np.array([[-np.exp(alpha[2]), np.exp(alpha[2])], [np.exp(alpha[3]), -np.exp(alpha[3])]])
    # create probablity to observe 1
    p0 = 1./(1 + np.exp(beta[0]))
    p1 = 1./(1 + np.exp(beta[1]))
    n_obs = obs.shape[0]
    # initialization P(s0, o0)
    n_state = 2
    p = np.array([np.log(1) + np.log(prob_obs(p0, obs[0])), np.log(0) + np.log(prob_obs(p1, obs[0]))])
    # Forward
    for s in range(n_obs - 1):
        if t[s+1] < 5: 
            T = expm(Q0*(t[s+1] - t[s]))
        if t[s] < 5 and t[s+1] > 5:
            T = np.matmul(expm(Q0*(5-t[s])), expm(Q1*(t[s+1]-5)))
        if t[s] > 5:
            T = expm(Q1*(t[s+1] - t[s]))
        p_obs = np.array([np.log(prob_obs(p0, obs[s+1])), np.log(prob_obs(p1, obs[s+1]))])
        P = np.log(T) + np.tile(p.reshape(-1, 1), [1, n_state]) + np.tile(p_obs.reshape(1, -1), [n_state, 1]) 
        # import pdb
        # pdb.set_trace()
        p = np.log(np.sum(np.exp(P), axis = 0))
    res = np.sum(np.exp(p))
    return np.log(res)

def Viterbi(alpha, beta, t, obs):
    """
    Viterbi algorithm to compute the optimal state sequence
    """
    # create transition intensity matrix
    Q0 = np.array([[-np.exp(alpha[0]), np.exp(alpha[0])], [np.exp(alpha[1]), -np.exp(alpha[1])]])
    Q1 = np.array([[-np.exp(alpha[2]), np.exp(alpha[2])], [np.exp(alpha[3]), -np.exp(alpha[3])]])
    # create probablity to observe 1
    p0 = 1./(1 + np.exp(beta[0]))
    p1 = 1./(1 + np.exp(beta[1]))
    n_obs = obs.shape[0]
    # initialization
    n_state = 2
    T1 = np.zeros([n_state, n_obs])
    T2 = np.zeros([n_state, n_obs])
    T1[0, 0] = np.log(prob_obs(p0, obs[0]))
    T1[1, 0] = np.log(0.)
    for s in range(n_obs-1):
        if t[s+1] < 5: 
            T = expm(Q0*(t[s+1] - t[s]))
        if t[s] < 5 and t[s+1] > 5:
            T = np.matmul(expm(Q0*(5-t[s])), expm(Q1*(t[s+1]-5)))
        if t[s] > 5:
            T = expm(Q1*(t[s+1] - t[s]))
        p_obs = np.array([np.log(prob_obs(p0, obs[s+1])), np.log(prob_obs(p1, obs[s+1]))])
        P = np.tile(T1[:, s].reshape(-1, 1), [1, n_state]) + np.log(T) + np.tile(p_obs.reshape(1, -1), [n_state, 1]) 
        T1[:, s+1] = np.max(P, axis = 0)
        T2[:, s+1] = np.argmax(P, axis = 0)
    # import pdb
    # pdb.set_trace()
    last_state = np.argmax(T1[:, -1])
    inverse_states = [last_state]
    for s in range(n_obs-1):
        curr_state = inverse_states[-1]
        inverse_states.append(T2[int(curr_state), -s-1])
    # import pdb
    # pdb.set_trace()
    return np.asarray(inverse_states[::-1]).astype(int)

def LLK(alpha, beta, states, t, obs, prior_z = 0.5):
    # create transition intensity matrix
    Q0 = np.array([[-np.exp(alpha[0]), np.exp(alpha[0])], [np.exp(alpha[1]), -np.exp(alpha[1])]])
    Q1 = np.array([[-np.exp(alpha[2]), np.exp(alpha[2])], [np.exp(alpha[3]), -np.exp(alpha[3])]])
    # create probablity to observe 1
    p0 = 1./(1 + np.exp(beta[0]))
    p1 = 1./(1 + np.exp(beta[1]))
    n_obs = obs.shape[0]
    # 
    res = np.log(prior_z)
    res += np.log(prob_obs(p0, obs[0]))
    for s in range(n_obs-1):
        if t[s+1] < 5: 
            T = expm(Q0*(t[s+1] - t[s]))
        if t[s] < 5 and t[s+1] > 5:
            T = np.matmul(expm(Q0*(5-t[s])), expm(Q1*(t[s+1]-5)))
        if t[s] > 5:
            T = expm(Q1*(t[s+1] - t[s]))
        p_obs = np.array([np.log(prob_obs(p0, obs[s+1])), np.log(prob_obs(p1, obs[s+1]))])
        res += np.log(T[states[s], states[s+1]]) + p_obs[states[s+1]]
    return res

def LLK_tran(alpha, states, t):
    # create transition intensity matrix
    Q0 = np.array([[-np.exp(alpha[0]), np.exp(alpha[0])], [np.exp(alpha[1]), -np.exp(alpha[1])]])
    Q1 = np.array([[-np.exp(alpha[2]), np.exp(alpha[2])], [np.exp(alpha[3]), -np.exp(alpha[3])]])
    # create probablity to observe 1
    n_obs = t.shape[0]
    res = 0
    for s in range(n_obs-1):
        if t[s+1] < 5: 
            T = expm(Q0*(t[s+1] - t[s]))
        if t[s] < 5 and t[s+1] > 5:
            T = np.matmul(expm(Q0*(5-t[s])), expm(Q1*(t[s+1]-5)))
        if t[s] > 5:
            T = expm(Q1*(t[s+1] - t[s]))
        res += np.log(T[states[s], states[s+1]])
    return res

def NEMCLL(pars, pos_z, opt_seqs, time, observations, prior_z):
    # pars includes alpha and beta
    alpha0, alpha1, beta = pars[:4], pars[4:8], pars[8:]
    res = 0
    N = time.shape[0]
    res = 0
    for n in range(N):
        # for index in range(2):
        #     if index == 0:
        #         llk = LLK(alpha0, beta, opt_seqs[n,0,:], time[n], observations[n])
        #     else:
        #         llk = LLK(alpha1, beta, opt_seqs[n,1,:], time[n], observations[n])
        #    res += pos_z[n, index] * llk
        res += pos_z[n, 0]*LLK(alpha0, beta, opt_seqs[n,0,:], time[n], observations[n], prior_z[0]) + pos_z[n, 1]*LLK(alpha1, beta, opt_seqs[n,1,:], time[n], observations[n], prior_z[1])
    return -res

def NEMCLL_tran(pars, pos_z, opt_seqs, time, sample_size=None):
    alpha0, alpha1 = pars[:4], pars[4:]
    res = 0
    N = time.shape[0]
    if sample_size is None:
        indexes = np.arange(N)
    else:
        indexes = np.random.choice(N, sample_size)
    for n in indexes:
        res += pos_z[n, 0]*LLK_tran(alpha0, opt_seqs[n,0,:], time[n]) + pos_z[n, 1]*LLK_tran(alpha1, opt_seqs[n,1,:], time[n]) 
    return -res

def NEMCLL_tran_est(pos_z, opt_seqs, time):
    N = opt_seqs.shape[0]
    # estimate transition rate for model z
    alpha0 = np.zeros(4)
    alpha1 = np.zeros(4)
    for z in range(2):
        tm00, tw00 = list(), list()
        tm01, tw01 = list(), list()
        tm10, tw10 = list(), list()
        tm11, tw11 = list(), list()
        for n in range(N):
            t_start, s_start = time[n,0], opt_seqs[n,z,0]
            # import pdb
            # pdb.set_trace()
            for t, s in zip(time[n], opt_seqs[n,z,:]):
                if s != s_start:
                    if t<5 and t_start<5:
                        if s_start == 0:
                            tm00.append(t-t_start)
                            tw00.append(pos_z[n, z]) 
                        if s_start == 1:
                            tm01.append(t-t_start)
                            tw01.append(pos_z[n, z])
                    if t>5 and t_start>5:
                        if s_start == 0:
                            tm10.append(t-t_start)
                            tw10.append(pos_z[n, z]) 
                        if s_start == 1:
                            tm11.append(t-t_start)
                            tw11.append(pos_z[n, z])
                    s_start = s
                    t_start = t
            # import pdb
            # pdb.set_trace()
        tw00 = np.array(tw00)
        tw00 /= np.sum(tw00)
        tw01 = np.array(tw01)
        tw01 /= np.sum(tw01)
        tw10 = np.array(tw10)
        tw10 /= np.sum(tw10)
        tw11 = np.array(tw11)
        tw11 /= np.sum(tw11)
        # print(tm00)
        # print(tm01)
        # print(tm10)
        # print(tm11)
        if z == 0:
            alpha0[0] = 1./np.dot(np.array(tm00),tw00)
            alpha0[1] = 1./np.dot(np.array(tm01),tw01)
            alpha0[2] = 1./np.dot(np.array(tm10),tw10)
            alpha0[3] = 1./np.dot(np.array(tm11),tw11)
        if z == 1:
            alpha1[0] = 1./np.dot(np.array(tm00),tw00)
            alpha1[1] = 1./np.dot(np.array(tm01),tw01)
            alpha1[2] = 1./np.dot(np.array(tm10),tw10)
            alpha1[3] = 1./np.dot(np.array(tm11),tw11)
    return np.concatenate([alpha0, alpha1])

def NEMCLL_emis_est(pos_z, opt_seqs, observations):
    N = opt_seqs.shape[0]
    n_state = 2
    n_obs = 2
    n_table = np.ones([n_state, n_obs])
    for n in range(N):
        for s0, s1, o in zip(opt_seqs[n,0,:], opt_seqs[n,1,:], observations[n]):
            n_table[s0, o] += pos_z[n, 0]
            n_table[s1, o] += pos_z[n, 1]
    print(n_table)
    freq_table = n_table/np.sum(n_table, axis = 1).reshape([-1,1])
    print(freq_table)
    
    ps = freq_table[:,1]
    beta = np.log(1/ps - 1)
    return beta

if __name__ == "__main__":
    # load data
    # with open("synthetic_data1.pickle", "rb") as res:
    #     time, observations, true_states = pickle.load(res)
    with open("synthetic_data2.pickle", "rb") as res:
        time, observations, true_states = pickle.load(res)
    
    # boostrapt
    indexes = np.random.choice(time.shape[0], time.shape[0])
    time = time[indexes]
    observations = observations[indexes]
    true_states = true_states[indexes]    

    # initialisation
    N = time.shape[0]
    # informative
    alpha0 = -np.array([1,1,0,0])
    alpha1 = -np.array([0,0,1,1])
    # non-informative
    # alpha0 = np.zeros(4)
    # alpha1 = np.ones(4)
    # true
    # alpha0 = np.array([-1.61, -1.61, 0.69, 0.69])
    # alpha1 = np.array([0.69, 0.69, -1.61, -1.61])
    # informative
    beta0 = -1
    beta1 = 1
    # noninformative
    # beta0 = np.random.randn()
    # beta1 = np.random.randn()
    # true
    # beta0 = -2.197
    #beta1 = 2.197
    
    beta = np.array([beta0, beta1])
    alpha = np.concatenate([alpha0, alpha1])
    # pars = np.concatenate([alpha, beta])
    
    n_structure = 2
    T = 50
    pi_z = np.array([0.5, 0.5])
    n_replication = 1
    sample_size = None
    

    # EM
    for rep in range(n_replication):
        for t in range(T):
            # compute marginal log likelihood
            mllk = 0
            for n in range(N):
                p_z = pi_z * np.exp(np.array([FFBS(alpha0, beta, time[n], observations[n]), FFBS(alpha1, beta, time[n], observations[n])]))
                mllk += np.log(p_z.sum())
            print("marginal log likelihood:{}".format(mllk))

            # given alpha, compute the conditional posterior distribution of z
            pos_z = []
            for n in range(N):
                p_z = pi_z * np.exp(np.array([FFBS(alpha0, beta, time[n], observations[n]), FFBS(alpha1, beta, time[n], observations[n])]))
                pos_z.append(p_z/np.sum(p_z))
                # import pdb
                # pdb.set_trace()
            pos_z = np.stack(pos_z)
            print(np.sum(pos_z[:,0] > 0.5))
            print(pos_z[:,0] > 0.5)
            # ##### use the true pos_z 
            # pos_z = np.concatenate([np.tile(np.array([[1., 0.]]), [20, 1]), np.tile(np.array([[0., 1.]]), [80, 1])])
            # #####

            # update the optimal state sequence given observations and model indicator
            opt_seqs = []
            # import pdb
            # pdb.set_trace()
            for n in range(N):
                opt_seq_n = []
                for index in range(n_structure):
                    if index == 0:
                        opt_seq_n.append(Viterbi(alpha0, beta, time[n], observations[n]))
                    else:
                        opt_seq_n.append(Viterbi(alpha1, beta, time[n], observations[n]))
                opt_seqs.append(np.stack(opt_seq_n))
            opt_seqs = np.stack(opt_seqs)
            print(opt_seqs.shape)
            # import pdb
            # pdb.set_trace()

            # ##### use the true states
            # opt_seqs[:20, 0, :] = true_states[:20]
            # opt_seqs[20:, 1, :] = true_states[20:]
            # #####

            # maximize the expected marginal complete log likelihood
            ## res_M = minimize(NEMCLL, x0 = pars, args=(pos_z, opt_seqs, time, observations, pi_z), method="L-BFGS-B", options={'disp': True, 'maxiter': 5})
            ## pars = res_M.x
            ## alpha0, alpha1, beta = pars[:4], pars[4:8], pars[8:]

            # update transition parameters
            res_tran = minimize(NEMCLL_tran, x0 = alpha, args=(pos_z, opt_seqs, time, sample_size), method="L-BFGS-B", options={'disp': True, 'maxiter': 5})
            alpha = res_tran.x
            # fast update transition parameters
            # alpha = NEMCLL_tran_est(pos_z, opt_seqs, time)

            alpha0, alpha1 = alpha[:4], alpha[4:8]
            # update emission parameters
            beta = NEMCLL_emis_est(pos_z, opt_seqs, observations)
            # optimize emssion parameters
            print(np.exp(alpha0), np.exp(alpha1),  1./(1 + np.exp(beta)))

     
        # import pdb
        # pdb.set_trace()
        with open("res/res_EM_boostrapt_{}.pickle".format(rank), "wb") as res:
            pickle.dump([pos_z, opt_seqs, alpha, beta, res_tran.fun], res)


        # with open("res/res_EM1.pickle".format(rank, rep), "wb") as res:
        #    pickle.dump([pos_z, opt_seqs, pars, res_M.fun], res)

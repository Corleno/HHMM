import pickle
import numpy as np
from scipy.linalg import expm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, f1_score, average_precision_score, precision_score, recall_score, roc_auc_score


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
        T = transition_mat(t[s], t[s+1], Q0, Q1)
        p_obs = np.array([np.log(prob_obs(p0, obs[s+1])), np.log(prob_obs(p1, obs[s+1]))])
        P = np.log(T) + np.tile(p.reshape(-1, 1), [1, n_state]) + np.tile(p_obs.reshape(1, -1), [n_state, 1]) 
        # import pdb
        # pdb.set_trace()
        p = np.log(np.sum(np.exp(P), axis = 0))
    res = np.sum(np.exp(p))
    return np.log(res)

def FFBS_posterior_last_state(alpha, beta, t, obs):
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
        T = transition_mat(t[s], t[s+1], Q0, Q1)
        p_obs = np.array([np.log(prob_obs(p0, obs[s+1])), np.log(prob_obs(p1, obs[s+1]))])
        P = np.log(T) + np.tile(p.reshape(-1, 1), [1, n_state]) + np.tile(p_obs.reshape(1, -1), [n_state, 1]) 
        # import pdb
        # pdb.set_trace()
        p = np.log(np.sum(np.exp(P), axis = 0))
    res = np.exp(p)/np.sum(np.exp(p))
    return res

def transition_mat(t0, t1, Q0, Q1):
    if t1 < 5: 
        T = expm(Q0*(t1 - t0))
    if t0 < 5 and t1 > 5:
        T = np.matmul(expm(Q0*(5-t0)), expm(Q1*(t1-5)))
    if t0 > 5:
        T = expm(Q1*(t1 - t0))
    return T

def prob_obs(p, obs):
    return p*obs+(1-p)*(1-obs)

if __name__ == "__main__":
    # load data
    with open("synthetic_data_test.pickle", "rb") as res:
        time, observations, true_states = pickle.load(res)

    # load model results
    # with open("res_1/res_EM_full.pickle", "rb") as res:
    with open("res/res_EM_full.pickle", "rb") as res:
        pos_z, opt_seqs, alpha_vec, beta, _ = pickle.load(res)

    N = time.shape[0]
    n_structure = 2
    pi_z = np.array([0.5, 0.5])
    true_frailty = np.concatenate([np.zeros(500), np.ones(500)])
    alpha_mat = alpha_vec.reshape([2, -1])
    
    # compute conditional posterior distribution of z
    pos_zs = [] 
    for n in range(N):
        p_z = pi_z * np.exp(np.array([FFBS(alpha_mat[0], beta, time[n][:-1], observations[n][:-1]), FFBS(alpha_mat[1], beta, time[n][:-1], observations[n][:-1])]))
        pos_zs.append(p_z/np.sum(p_z))
    pos_zs = np.stack(pos_zs)
    print(confusion_matrix(true_frailty, (pos_zs[:,1] > 0.5).astype(int)))
    print(accuracy_score(true_frailty, (pos_zs[:,1] > 0.5).astype(int)))


    # compute posterior of the last state
    pos_last_states = list()
    for n in range(N):
        pos_cond_last_state = list()
        for z in range(n_structure):
            alpha = alpha_mat[z]
            pos_cond_second_last_state = FFBS_posterior_last_state(alpha, beta, time[n][:-1], observations[n][:-1])
            Q0 = np.array([[-np.exp(alpha[0]), np.exp(alpha[0])], [np.exp(alpha[1]), -np.exp(alpha[1])]])
            Q1 = np.array([[-np.exp(alpha[2]), np.exp(alpha[2])], [np.exp(alpha[3]), -np.exp(alpha[3])]])
            tran_mat = transition_mat(time[n][-2], time[n][-1], Q0, Q1)
            pos_cond_last_state.append(np.dot(pos_cond_second_last_state, tran_mat))
        pos_cond_last_state = np.stack(pos_cond_last_state)
        pos_last_state = np.dot(pos_cond_last_state.T, pos_zs[n,:])        
        pos_last_states.append(pos_last_state)
    pos_last_states = np.stack(pos_last_states)
    acc = accuracy_score(true_states[:,-1], np.round(pos_last_states[:,-1]))
    print("last_state: acc {}".format(acc))

    # import pdb
    # pdb.set_trace()

    # compute the predictive posterior of last observation
    post_last_observations = list()
    p0 = 1./(1 + np.exp(beta[0]))
    p1 = 1./(1 + np.exp(beta[1]))
    emis_mat = np.stack([np.array([1-p0, p0]), np.array([1-p1, p1])])
    for n in range(N):
        post_last_observations.append(np.dot(pos_last_states[n], emis_mat))
    post_last_observations=np.stack(post_last_observations)

    # import pdb
    # pdb.set_trace()

    # summarize prediction results
    
    acc = accuracy_score(observations[:,-1], np.round(post_last_observations[:,-1]))
    roc_auc = roc_auc_score(observations[:,-1], post_last_observations[:,-1])
    f1 = f1_score(observations[:,-1], np.round(post_last_observations[:,-1]))
    aver_prec = average_precision_score(observations[:,-1], np.round(post_last_observations[:,-1]))
    prec = precision_score(observations[:,-1], np.round(post_last_observations[:,-1]))
    recall = recall_score(observations[:,-1], np.round(post_last_observations[:,-1]))
    print("accuray: {}, auc: {}, f1: {}, average_prec: {}, prec: {}, recall: {}".format(acc, roc_auc, f1, aver_prec, prec, recall))
            
    import pdb
    pdb.set_trace()




    
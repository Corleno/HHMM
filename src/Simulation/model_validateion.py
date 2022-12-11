import pickle
import numpy as np 

if __name__ == "__main__":
    # load data
    with open("synthetic_data2.pickle", "rb") as res:
        time, observations, true_states = pickle.load(res)

    # load model results
    with open("res/res_EM_full.pickle", "rb") as res:
    	pos_z, opt_seqs, alpha, beta, _ = pickle.load(res)
    print(np.exp(alpha))
    print(1/(1+np.exp(beta)))

    # check the fraility
    true_frailty = np.concatenate([np.zeros(300), np.ones(200)])
    # import pdb
    # pdb.set_trace()
    pred_frailty = np.array([p[1]>0.5 for p in pos_z]).astype(np.int32)
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    cmat = confusion_matrix(true_frailty, pred_frailty)
    print(cmat)
    print('accuracy: {}'.format(accuracy_score(true_frailty, pred_frailty)))
    print('precision: {}'.format(precision_score(true_frailty, pred_frailty)))
    print('recall: {}'.format(recall_score(true_frailty, pred_frailty)))

    N = time.shape[0]
    pred_seqs = [seq[0] if p[0]>0.5 else seq[1] for p, seq in zip(pos_z, opt_seqs)]
    accuracy_list = list()
    precision_list = list()
    recall_list = list()
    for true_seq, pred_seq in zip(true_states, pred_seqs):
    	accuracy_list.append(accuracy_score(true_seq, pred_seq))
    	precision_list.append(precision_score(true_seq, pred_seq))
    	recall_list.append(recall_score(true_seq, pred_seq))
        # import pdb
        # pdb.set_trace()
    accuracy_scores = np.array(accuracy_list)
    precision_scores = np.array(precision_list)
    recall_scores = np.array(recall_list)
    print(accuracy_scores.mean(), accuracy_scores.std())
    print(precision_scores.mean(), precision_scores.std())
    print(recall_scores.mean(), recall_scores.std())

    # load boostrapt results
    alpha_list = list()
    beta_list = list()
    for i in range(100):
        with open("res/res_EM_boostrapt_{}.pickle".format(i), 'rb') as res:
        	pos_z, opt_seqs, alpha, beta, _ = pickle.load(res)
        print('alpha', np.exp(alpha))
        print('beta', 1/(1+np.exp(beta)))
        alpha_list.append(np.exp(alpha))
        beta_list.append(1/(1+np.exp(beta)))
    alpha_boostrapt = np.stack(alpha_list)
    beta_boostrapt = np.stack(beta_list)

    print("alpha-> mean: {}, std: {}".format(np.mean(alpha_boostrapt, axis=0), np.std(alpha_boostrapt, axis=0)))
    print("beta-> mean: {}, std: {}".format(np.mean(beta_boostrapt, axis=0), np.std(beta_boostrapt, axis=0)))

import numpy as np
import pickle

if __name__ == "__main__":
	# # load train data
    # with open("synthetic_data1.pickle", "rb") as res:
    #     train_time, train_observations, train_true_states = pickle.load(res)
    # with open("synthetic_data_test.pickle", "rb") as res:
    # 	test_time, test_observations, test_true_states = pickle.load(res)

    # X_train, y_train = np.stack([train_time[:, :-1], train_observations[:, :-1]], axis=1), train_observations[:, -1]
    # X_test, y_test = np.stack([test_time[:, :-1], test_observations[:, :-1]], axis=1), test_observations[:, -1]

    # with open("../../data/sim/vectorized_data.pickle", "wb") as res:
    #     pickle.dump([X_train, X_test, y_train, y_test], res)

    # load train data
    with open("synthetic_data2.pickle", "rb") as res:
    	train_time, train_observations, train_true_states = pickle.load(res)
    with open("synthetic_data_test.pickle", "rb") as res:
    	test_time, test_observations, test_true_states = pickle.load(res)

    X_train, y_train = np.stack([train_time[:, :-1], train_observations[:, :-1]], axis=1), train_observations[:, -1]
    X_test, y_test = np.stack([test_time[:, :-1], test_observations[:, :-1]], axis=1), test_observations[:, -1]

    with open("../../data/sim/vectorized_data2.pickle", "wb") as res:
    	pickle.dump([X_train, X_test, y_train, y_test], res)
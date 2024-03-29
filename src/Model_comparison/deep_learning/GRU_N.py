#!/usr/bin/python3
# date: 07/27/2019

# LSTM to predict the last visiting statue

import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.layers import GRU
from keras import optimizers

import matplotlib
from keras import optimizers

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time


def pad_zeros_mat(x, nfeature=None, maxlen=None):
    nr, nc = x.shape
    if not nfeature:
        nfeature = nr
    if not maxlen:
        maxlen = nc
    new_x = np.zeros([nfeature, maxlen])
    new_x[:nr, :nc] = x
    return new_x


def pad_zeros(x_list, maxlen=None):
    new_x_list = []
    for x in x_list:
        pad_x = pad_zeros_mat(x, maxlen=maxlen)
        new_x_list.append(pad_x.T)
    return np.stack(new_x_list)


def get_last_status(Y):
    s = np.zeros(len(Y))
    # import pdb
    # pdb.set_trace()
    for indx, patientY in enumerate(Y):
        patientY_last = patientY.T[-1]
        # print("patientY_last:", patientY_last)
        if patientY_last[[2, 3, 6, 7]].sum() > 0:
            patient_last_statue = 1
        else:
            patient_last_statue = 0
        s[indx] = patient_last_statue
    return s


# Data generator
class BatchGenerator(object):
    def __init__(self, data, label, batch_size, skip_step=5):
        self.data = data
        self.label = label
        self.curr_idx = 0
        self.skip_step = skip_step
        self.N, self.T_max, self.n_feature = data.shape
        self.batch_size = batch_size

    def generate(self):
        x = np.zeros([self.batch_size, self.T_max, self.n_feature])
        y = np.zeros([self.batch_size])
        while True:
            for i in range(self.batch_size):
                if self.curr_idx >= self.N:
                    self.curr_idx = 0
                x[i] = self.data[self.curr_idx]
                y[i] = self.label[self.curr_idx]
            self.curr_idx += self.skip_step
            yield x, y


class Model(object):
    def __init__(self):
        self.model = Sequential()
        self.hidden_size = args.size
        self.output_size = 10
        self.T_max = 43
        self.n_feature = 2

    def build_model(self):
        self.model.add(Masking(mask_value=0., input_shape=(self.T_max, self.n_feature)))
        self.model.add(GRU(self.hidden_size, return_sequences=True))
        self.model.add(Dense(self.output_size, activation="exponential"))
        # self.model.add(Activation(activation = "softmax"))
        # compile the model
        adam = optimizers.Adam(lr=0.01)
        self.model.compile(optimizer=adam, loss='poisson', metrics=['acc'])
        # summarize the model
        print(self.model.summary())

    def fit(self, X, Y, n_epoch=20, n_iter=10):
        # print(train_data_generator.generate())
        N, T_max, n_feature = X.shape
        batch_size = N // n_iter

        # Using personal defined generator
        # train_data_generator = BatchGenerator(X, Y, batch_size = batch_size)
        # # train model
        # for epoch in range(n_epoch):
        #     for iteration in range(n_iter):
        #         self.model.fit(*next(train_data_generator.generate()))

        # Using default fit
        self.model.fit(X, Y, batch_size=batch_size, epochs=n_epoch)

    def evaluate(self, X, Y, n_iter=10, confusion=False):
        N, T_max, n_feature = X.shape
        batch_size = N // n_iter
        # loss, acc = self.model.evaluate(X, Y, batch_size=batch_size)
        # print("loss: {}, acc: {}".format(loss, acc))
        if confusion:
            pred_Y_lambda = self.model.predict(X)
            pred_Y_lambda_last = pred_Y_lambda[:, -1, [2, 3, 6, 7]]
            pred_Y_last_normal = np.exp(-np.sum(pred_Y_lambda_last, axis=-1))
            pred_Y_last = (pred_Y_last_normal <= 0.5).astype(int)
            TP = (pred_Y_last[Y == 1] == 1).sum()
            FP = (pred_Y_last[Y == 1] == 0).sum()
            TN = (pred_Y_last[Y == 0] == 0).sum()
            FN = (pred_Y_last[Y == 0] == 1).sum()
            print("TP: {}, FP: {}, TN:{}, FN:{}".format(TP, FP, TN, FN))
            # model evaluation:
            from sklearn.metrics import roc_curve, auc, f1_score, average_precision_score, precision_score, \
                recall_score, roc_auc_score
            roc_auc = roc_auc_score(Y, 1-pred_Y_last_normal)
            f1 = f1_score(Y, pred_Y_last)
            aver_prec = average_precision_score(Y, pred_Y_last)
            prec = precision_score(Y, pred_Y_last)
            recall = recall_score(Y, pred_Y_last)
            print(
                "auc: {}, f1: {}, average_prec: {}, prec: {}, recall: {}".format(roc_auc, f1, aver_prec, prec, recall))

            # auc curve
            fpr, tpr, thresholds = roc_curve(Y, pred_Y_last)
            fig = plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC_gru')
            plt.legend(loc="lower right")
            plt.savefig("../../res/ROC_gru.png")
            plt.close(fig)

    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model_gru.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model_gru.h5")
        print("Saved model to disk")

    def load_model(self):
        from keras.models import model_from_json
        # load json and create model
        json_file = open('model_gru.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model_gru.h5")
        self.model = loaded_model
        print("Loaded model from disk")
        # compile the model
        self.model.compile(optimizer='adam', loss='poisson', metrics=['acc'])
        # summarize the model
        print(self.model.summary())

    def predict(self, X):
        pred_Y = self.model.predict(X)
        print("pred_Y: {}".format(pred_Y))
        # probabilities to be 1
        return pred_Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', help="layer size", type=int, default=16)
    args = parser.parse_args()
    with open("../../data/data_1000/vectorized_data_trainandtest_N.pickle", "rb") as res:
        X_train, X_test, Y_train, Y_test = pickle.load(res)

    # padding data
    maxlen_train = np.max([X.shape[1] for X in X_train])
    print("maximum length: {}".format(maxlen_train))
    maxlen_test = np.max([X.shape[1] for X in X_test])
    print("maximum length: {}".format(maxlen_test))

    maxlen = max(maxlen_train, maxlen_test)

    # padding inputs
    pad_features_train = pad_zeros(X_train, maxlen)
    print(pad_features_train.shape)
    pad_features_test = pad_zeros(X_test, maxlen)
    print(pad_features_test.shape)
    # padding outputs
    pad_output_train = pad_zeros(Y_train, maxlen)
    print(pad_output_train.shape)
    pad_output_test = pad_zeros(Y_test, maxlen)
    print(pad_output_test.shape)
    output_test = get_last_status(Y_test)
    print(output_test.shape)

    ts = time.time()
    GRU_model = Model()
    GRU_model.build_model()
    GRU_model.fit(pad_features_train, pad_output_train, n_epoch=200, n_iter=5)
    GRU_model.save_model()
    print("training time: {}s".format(time.time() - ts))

    ts = time.time()
    GRU_model.load_model()
    GRU_model.evaluate(pad_features_test, output_test, confusion=True)
    print("testing time: {}s".format(time.time() - ts))
    # GRU_model.predict(pad_features_test)
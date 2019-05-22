#!/usr/bin/python3
# author: Rui Meng
# date: 05/03/2019

# LSTM to predict the last visiting statue

import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Masking, Activation
from keras.layers import LSTM
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import numpy as np

def pad_zeros_mat(x, nfeature = None, maxlen = None):
    nr, nc = x.shape
    if not nfeature:
        nfeature = nr
    if not maxlen:
        maxlen = nc
    new_x = np.zeros([nfeature, maxlen])
    new_x[:nr, :nc] = x
    return new_x

def pad_zeros(x_list, maxlen = None):
    new_x_list = []
    for x in x_list:
        pad_x = pad_zeros_mat(x, maxlen=maxlen)
        new_x_list.append(pad_x.T)
    return np.stack(new_x_list)

# Data generator
class BatchGenerator(object):
    def __init__(self, data, label, batch_size, skip_step = 5):
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
        self.hidden_size = 10
        self.T_max = 42
        self.n_feature = 12

    def build_model(self):
        self.model.add(Masking(mask_value=0., input_shape = (self.T_max, self.n_feature)))
        self.model.add(LSTM(self.hidden_size,  return_sequences=False))
        self.model.add(Dense(1, activation = "sigmoid"))
        # self.model.add(Activation(activation = "softmax"))
        # compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # summarize the model 
        print(self.model.summary())

    def fit(self, X, Y, n_epoch = 20, n_iter = 10):
        # print(train_data_generator.generate())
        N, T_max, n_feature = X.shape
        batch_size = N//n_iter

        # Using personal defined generator
        # train_data_generator = BatchGenerator(X, Y, batch_size = batch_size)
        # # train model
        # for epoch in range(n_epoch):
        #     for iteration in range(n_iter):
        #         self.model.fit(*next(train_data_generator.generate()))
        
        # Using default fit
        self.model.fit(X, Y, batch_size = batch_size, epochs = n_epoch)

    def evaluate(self, X, Y, n_iter=10, confusion=False):
        N, T_max, n_feature = X.shape
        batch_size = N//n_iter        
        loss, acc = self.model.evaluate(X, Y, batch_size = batch_size)
        print("loss: {}, acc: {}".format(loss, acc))
        if confusion:
            pred_Y_score = self.model.predict(X)
            pred_Y = np.round(pred_Y_score)
            TP = (pred_Y[Y==1] == 1).sum()
            FP = (pred_Y[Y==1] == 0).sum()
            TN = (pred_Y[Y==0] == 0).sum()
            FN = (pred_Y[Y==0] == 1).sum()
            print("TP: {}, FP: {}, TN:{}, FN:{}".format(TP, FP, TN, FN))
            # model evaluation:
            from sklearn.metrics import roc_curve, auc, f1_score, average_precision_score, precision_score, recall_score, roc_auc_score
            roc_auc = roc_auc_score(Y, pred_Y_score)
            f1 = f1_score(Y, pred_Y)
            aver_prec = average_precision_score(Y, pred_Y)
            prec = precision_score(Y, pred_Y)
            recall = recall_score(Y, pred_Y)
            print("auc: {}, f1: {}, average_prec: {}, prec: {}, recall: {}".format(roc_auc, f1, aver_prec, prec, recall))
            
            # auc curve
            fpr, tpr, thresholds = roc_curve(Y, pred_Y)
            fig = plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC_lstm')
            plt.legend(loc="lower right")
            plt.savefig("../res/ROC_lstm.png")
            plt.close(fig)

    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model_lstm.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model_lstm.h5")
        print("Saved model to disk")

    def load_model(self):
        from keras.models import model_from_json
        # load json and create model
        json_file = open('model_lstm.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model_lstm.h5")
        self.model = loaded_model
        # compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # summarize the model 
        print(self.model.summary())

        print("Loaded model from disk")

    def predict(self, X):
        pred_Y = self.model.predict(X)
        print ("pred_Y: {}".format(pred_Y))
        # probabilities to be 1


if __name__ == "__main__":

    with open("../data/data_1000/vectorized_data_trainandtest.pickle", "rb") as res:
        X_train, X_test, y_train, y_test = pickle.load(res)

    # padding data
    maxlen_train = np.max([X.shape[1] for X in X_train])
    print("maximum length: {}".format(maxlen_train))
    maxlen_test = np.max([X.shape[1] for X in X_test])
    print("maximum length: {}".format(maxlen_test))

    maxlen = max(maxlen_train, maxlen_test)

    pad_features_train = pad_zeros(X_train, maxlen)
    print(pad_features_train.shape)
    pad_features_test = pad_zeros(X_test, maxlen)
    print(pad_features_test.shape)
    
    
    LSTM_model = Model()

    # LSTM_model.build_model()
    # LSTM_model.fit(pad_features_train, y_train, n_epoch = 1000, n_iter = 5)
    # LSTM_model.save_model()   


    LSTM_model.load_model()
    LSTM_model.evaluate(pad_features_test, y_test, confusion=True)
    
    # LSTM_model.predict(pad_features_test)
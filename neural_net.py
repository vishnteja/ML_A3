import numpy as np
import os
from PIL import Image
from scipy import misc
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

files = os.listdir("train")
# os.makedirs("inp_train")
# for file in files:
#     img = Image.open('train/' + file).convert('LA')
#     new_file = file[:3] + "_" + file[4:-3]
#     img.resize((50, 50)).save('inp_train/' + new_file+'.png')

files = os.listdir('inp_train')
x = []
for file in files:
    img = misc.imread('inp_train/' + file, flatten=True)
    x.append(img.flatten())
files = os.listdir("test")
x = np.array(x)
# os.makedirs("inp_test")
# for file in files:
#     img = Image.open('test/' + file).convert('LA')
#     new_file = file[:3] + "_" + file[4:-3]
#     img.resize((50, 50)).save('inp_test/' + new_file+'.png')

files = os.listdir('inp_test')
y = []
for file in files:
    img = misc.imread('inp_test/' + file, flatten=True)
    y.append(img.flatten())
y = np.array(y)


def a_s(a, b):
    return (0.51034)


labels_train = np.array([0]*11075 + [1]*11000)
one_hot_encoding_x = np.zeros((22075, 2))
for i in range(22075):
    one_hot_encoding_x[i, labels_train[i]] = 1
labels_test = np.array([0]*1425 + [1]*1500)
one_hot_encoding_y = np.zeros((2925, 2))
for i in range(2925):
    one_hot_encoding_y[i, labels_test[i]] = 1

print('Finished Preprocessing')


class Network:

    def __init__(self, no_h_layers, feature_set, labels_train, test, labels_test):
        self.labels_train = labels_train
        self.test = test
        self.labels_test = labels_test
        self.feature_set = feature_set
        self.no_hidden_layers = no_h_layers
        self.no_input_nodes = feature_set.shape[1]
        self.no_output_nodes = 2
        self.no_hidden_nodes = 5
        self.weights = []
        self.biases = []
        self.result = np.zeros(self.no_output_nodes)
        self.tempz = []
        self.tempa = []
        self.dcost_dw = []
        self.dcost_db = []
        self.alpha = 10e-4
        self.error_cost = []
        self.true = []
        self.pred = []
        self.log_cost = []
        self.initialize()

    def initialize(self):
        self.weights.append(np.random.rand(
            self.no_input_nodes, self.no_hidden_nodes))
        self.biases.append(np.random.randn(self.no_hidden_nodes))
        for i in range(self.no_hidden_layers-1):
            self.biases.append(np.random.randn(self.no_hidden_nodes))
            self.weights.append(np.random.rand(
                self.no_hidden_nodes, self.no_hidden_nodes))
        self.weights.append(np.random.rand(
            self.no_hidden_nodes, self.no_output_nodes))
        self.biases.append(np.random.randn(self.no_output_nodes))

    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def softmax(self, A):
        b = A.max()
        expA = np.exp(A - b)
        return expA / expA.sum(axis=1, keepdims=True)

    def feedforward(self):
        self.tempz.clear()
        self.tempa.clear()
        print("feedforward")
        self.tempz.append(
            np.dot(self.feature_set, self.weights[0]) + self.biases[0])
        self.tempa.append(self.sigmoid(self.tempz[0]))
        for i in range(self.no_hidden_layers-1):
            print(i)
            self.tempz.append(
                np.dot(self.tempa[i], self.weights[i+1]) + self.biases[i+1])
            self.tempa.append(self.sigmoid(self.tempz[-1]))
        self.result = self.softmax(
            np.dot(self.tempa[-1], self.weights[-1]) + self.biases[-1])

    def backpropogate(self):
        self.dcost_dw = []
        self.dcost_db = []
        print("backpropogate")
        dcost_dzo = self.result - self.labels_train
        dzo_dwo = self.tempa[-1]
        self.dcost_dw.append(np.dot(dzo_dwo.T, dcost_dzo))
        self.dcost_db.append(dcost_dzo)

        dzo_dah = self.weights[-1]
        for i in range(len(self.weights)-2, 0, -1):
            dzh_dwh = self.tempa[i]
            dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
            dah_dzh = self.derivative_sigmoid(self.tempz[i])
            self.dcost_dw.append(np.dot(dzh_dwh.T, dah_dzh * dcost_dah))
            self.dcost_db.append(dcost_dah * dah_dzh)
        dzh_dwh = self.feature_set
        dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
        dah_dzh = self.derivative_sigmoid(self.tempz[0])
        self.dcost_dw.append(np.dot(dzh_dwh.T, dah_dzh * dcost_dah))
        self.dcost_db.append(dcost_dah * dah_dzh)
        self.dcost_dw = list(reversed(self.dcost_dw))
        self.dcost_db = list(reversed(self.dcost_db))

    def update_weights(self):
        print("update_weights")
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * self.dcost_dw[i]
            self.biases[i] -= self.alpha * self.dcost_db[i].sum(axis=0)

    def train(self):
        for epoch in range(2):
            self.feedforward()
            self.backpropogate()
            self.update_weights()
            if epoch % 1 == 0:
                loss = -1*np.sum(self.labels_train * np.log(self.result))
                self.error_cost.append(loss)
        print(self.error_cost)

    def testmodel(self):
        self.tempz.clear()
        self.tempa.clear()
        self.tempz.append(np.dot(self.test, self.weights[0]) + self.biases[0])
        self.tempa.append(self.sigmoid(self.tempz[0]))
        for i in range(self.no_hidden_layers-1):
            self.tempz.append(
                np.dot(self.tempa[i], self.weights[i+1]) + self.biases[i+1])
            self.tempa.append(self.sigmoid(self.tempz[-1]))
        self.result = self.softmax(
            np.dot(self.tempa[-1], self.weights[-1]) + self.biases[-1])

    def modify(self):
        for item in self.result:
            if item[0] > item[1]:
                self.pred.append(0)
            else:
                self.pred.append(1)

        for item in self.labels_test:
            if item[0] > item[1]:
                self.true.append(0)
            else:
                self.true.append(1)

    def print_results(self):
        # print(self.result)
        # print(self.labels_test)
        self.result = self.pred
        self.labels_test = self.true
        # print(self.result)
        # print(self.labels_test)
        print("\nAccuracy: " + str(accuracy_score(self.labels_test, self.result)))

        # conf_matrix = confusion_matrix(
        #     self.labels_test, self.result)
        # print("\nConfusion Matrix: ")
        # print("               Predicted")
        # print("               pos   neg")
        # print("Actual pos    ", end="")
        # print(conf_matrix[0])
        # print("       neg    ", end="")
        # print(conf_matrix[1])
        # print("\nPrecision")
        # print("pos: " + str(precision_score(self.labels_test,
        #                                     self.result, pos_label="pos")))
        # print("neg: " + str(precision_score(self.labels_test,
        #                                     self.result, pos_label="neg")))

        # print("\nRecall")
        # print("pos: " + str(recall_score(self.labels_test, self.result, pos_label="pos")))
        # print("neg: " + str(recall_score(self.labels_test, self.result, pos_label="neg")))

        # print("\nF1 Measure")
        # print("pos: " + str(f1_score(self.labels_test, self.result, pos_label="pos")))
        # print("neg: " + str(f1_score(self.labels_test, self.result, pos_label="neg")))


nobject = Network(3, x, one_hot_encoding_x, y, one_hot_encoding_y)
nobject.train()
nobject.testmodel()
nobject.modify()
nobject.print_results()

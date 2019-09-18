import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
from math import exp
from random import seed
from random import random

# Initialize a network


def initialize(n_inputs, n_hidden, n_outputs):
    network = list()
    inp = n_inputs
    for k in n_hidden:
        hidden_layer = []
        hidden_layer = [{'weights': [random() for i in range(inp + 1)]}
                        for i in range(k)]
        inp = k
        network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden[-1] + 1)]}
                    for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def transfer_derivative(output):
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = np.zeros(n_outputs)
            expected[int(row[-1])] = 1
            sum_error += np.sum((expected-outputs)**2)
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def rgb2gray(rgb):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def load_images(folder):
    images = []
    cat = 0
    dog = 0
    for filename in os.listdir(folder):

        if 'cat' in filename:
            cat += 1
            if(cat > 10):
                continue
        else:
            dog += 1
            if(dog > 10):
                continue

        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (50, 50))
        img = rgb2gray(img)
        img = img.flatten()
        if 'cat' in filename:
            img = np.append(img, 0)
        else:
            if 'dog' in filename:
                img = np.append(img, 1)
        if img is not None:
            images.append(img)
    return images


dataset = load_images("train")

# dataset1 = [[2.7810836,2.550537003,0],
#	[1.465489372,2.362125076,0],
#	[3.396561688,4.400293529,0],
#	[1.38807019,1.850220317,0],
#	[3.06407232,3.005305973,0],
#	[7.627531214,2.759262235,1],
#	[5.332441248,2.088626775,1],
#	[6.922596716,1.77106367,1],
#	[8.675418651,-0.242068655,1],
#	[7.673756466,3.508563011,1]]
#n_inputs = len(dataset1[0]) - 1
#n_outputs = len(set([row[-1] for row in dataset1]))
#network = initialize(n_inputs, [2], n_outputs)
#train_network(network, dataset1, 0.5, 20, n_outputs)
# for layer in network:
#	print(layer)
# for row in dataset1:
#	prediction = predict(network, row)
#	print('Expected=%d, Got=%d' % (row[-1], prediction))

#img = cv2.imread('train//cat.0.jpg')
# img=cv2.resize(img,(50,50))
# img=rgb2gray(img)
# img=img.flatten()
# img=np.append(img,0)


n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize(n_inputs, [100], n_outputs)

train_network(network, dataset, 0.5, 5, n_outputs)
for layer in network:
    print(layer)

for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))

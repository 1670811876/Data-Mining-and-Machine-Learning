#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# Create data set

num_input = 1  # dimension
num_examples = 1000  # the number of samples generated
true_theta = np.array([[4.2, 2]])  # real parameter values, y=4.2+2x

X = np.random.randn(num_examples, num_input)  # the random function produces the column vector X
X = np.insert(X, 0, values=np.ones(num_examples), axis=1)  # insert one column

y = np.dot(X, true_theta.T)  # calculate the value y
y += 0.01 * np.random.randn(num_examples, num_input)  # add some noise

print(X[0], y[0])

plt.scatter(X[0:50, 1], y[0:50])
plt.show()


# Define the model

def net(X, theta):
    return np.dot(X, theta.T)


# Define cost function

def compute_cost(h, y):
    cost = np.power((h - y), 2)
    return np.sum(cost) / (2 * len(X))


# Gradient Descent Algorithm
'''
params:
X - training set
y - marking of training set
theta - model parameter
alpha - learning rate
iter - iterations
returns:
returns the solved model parameter theta
'''


def gradient_descent(X, y, theta, alpha, iter_num):
    temp = np.zeros(theta.shape)
    # ravel() converts a multidimensional array to a one-dimensional array
    para_num = int(np.ravel(theta).shape[0])  # get the number of parameters

    # Iterate to solve the parameters
    for i in range(iter_num):
        # get the error
        error = np.dot(X, theta.T) - y
        # calculate the gradient
        for j in range(para_num):
            term = error * X[:, j].reshape(error.shape)
            temp[0, j] = theta[0, j] - (alpha / len(X)) * np.sum(term)

        # update the parameter
        theta = temp
        # calculate the predicted value
        h = np.dot(X, theta.T)
        # calculate the loss CostJ
        loss = compute_cost(h, y)
        cost_list.append(loss)
        if (i + 1) % 100 == 0:
            print("Epoch %s. Moving loss: %s. theta0: %s. theta1: %s" % (i, loss, theta[0][0], theta[0][1]))
            plot(cost_list, X, theta)
    return theta


# Initialize model

theta = np.array([[0, 0]])


# Training

# model function
def real_fn(X):
    return 4.2 * X[:, 0] + 2 * X[:, 1]


# plot the line chart that losses lowers with the number of training increasing,
# and scatter diagram of predicted values and true values
def plot(losses, X, theta):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title("Loss during training")
    fg1.plot(xs, losses, '-r')
    fg2.set_title("Estimated cv real function")
    fg2.plot(X[:, 1], net(X, theta), 'or', label='Estimated')
    fg2.plot(X[:, 1], real_fn(X), '*g', label='real')
    fg2.legend()
    plt.show()


# Initialize the model parameters
theta = np.array([[0, 0]])
alpha = 0.01
iter_num = 500
cost_list = []
theta = gradient_descent(X, y, theta, alpha, iter_num)


# Draw the fitting curve

# return evenly spaced 100 numbers within the specified interval
x = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
# get predicted values using model
f = theta[0, 0] + (theta[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(X[:, 1], y, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Sample Regression")
plt.show()


# Draw the training error diagram
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iter_num), cost_list, 'r')
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
ax.set_title("Error VS Training Epoch")
plt.show()

print(theta)
print(true_theta)

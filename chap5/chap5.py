#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

num_inputs = 2  # the number pf parameters
num_examples = 1000  # the size of data set
true_theta = np.array([[4.2, 2, -3.4]])  # the true parameters which produce data
X = np.random.randn(num_examples, num_inputs)  # generate [0, 1) data according to given dimension
X = np.insert(X, 0, values=np.ones(num_examples), axis=1)  # insert a constant sequence in the first column

y = np.dot(X, true_theta.T)
y += 0.01 * np.random.randn(num_examples, 1)

print(X[0], y[0])

# Plot a scatter plot
plt.scatter(X[:, 2], y)
plt.show()


# Define model
def net(X, theta):
    return np.dot(X, theta.T)


# Define loss function
def compute_cost(h, y):
    cost = np.power((h - y), 2)
    return np.sum(cost) / (2 * len(X))


# Gradient Descent Algorithm
def gradient_descent(X, y, theta, alpha, iters):
    temp = np.zeros(theta.shape)
    cost = np.zeros(iters)
    para_num = int(np.ravel(theta).shape[0])

    # while true
    for i in range(iters):
        error = np.dot(X, theta.T) - y
        # calculate the gradient
        for j in range(para_num):
            term = error * X[:, j].reshape(error.shape)
            temp[0, j] = theta[0, j] - (alpha / len(X)) * np.sum(term)
        theta = temp
        h = net(X, theta)
        loss = compute_cost(h, y)
        costJ.append(loss)
        if (i + 1) % 100 == 0:
            print("Epoch %s. Moving loss: %s. theta0:%s, theta1:%s, theta2:%s"
                  % (i, loss, theta[0][0], theta[0][1], theta[0][2]))
            plot(costJ, X, theta)
    return theta


# Initialize model

theta = np.array([[0, 0, 0]])


# Training

# model function
def real_fn(X):
    return 4.2 * X[:, 0] + 2 * X[:, 1] - 3.4 * X[:, 2]


# Plot the line chart that losses lowers with the number of training increasing,
# and scatter diagram of predicted values and true values
def plot(losses, X, theta, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1],
             net(X[:sample_size, :], theta), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1],
             real_fn(X[:sample_size, :]), '*g', label='Real')
    fg2.legend()
    plt.show()


# Initialize model and train it
alpha = 0.01
theta = np.array([[0, 0, 0]])
iters = 500
costJ = []
theta = gradient_descent(X, y, theta, alpha, iters)

# Draw the training error diagram
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), costJ, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error Vs. Training Epoch')
plt.show()

print(theta)
print(true_theta)

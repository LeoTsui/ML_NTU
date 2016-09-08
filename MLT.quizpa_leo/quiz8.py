#!user/bin/env python3
# _*_ coding: utf-8 _*_

"""
Question 11
Experiment with Backprop neural Network

Implement the backpropagation algorithm (page 16 of lecture 212) for d-M-1 neural network with tanh-type neurons, including the output neuron. Use the squared error measure between the output gNNET(xn) and the desired yn and backprop to calculate the per-example gradient. Because of the different output neuron, your δ(L)1 would be different from the course slides! Run the algorithm on the following set for training (each row represents a pair of (xn,yn); the first column is (xn)1; the second one is (xn)2; the third one is yn):
hw4_nnet_train.dat
and the following set for testing:
hw4_nnet_test.dat

Fix T=50000 and consider the combinations of the following parameters:
	* the number of hidden neurons M
	* the elements of w(ℓ)ij chosen independently and uniformly from the range (−r,r)
	* the learning rate η

Fix η=0.1 and r=0.1. Then, consider M∈{1,6,11,16,21} and repeat the experiment for 500 times. Which M results in the lowest average Eout over 500 experiments?

6

Question 12
Following Question 11, fix η=0.1 and M=3. Then, consider r∈{0,0.1,10,100,1000} and repeat the experiment for 500 times. Which r results in the lowest average Eout over 500 experiments?

0.001

Question 13
Following Question 11, fix r=0.1 and M=3. Then, consider η∈{0.001,0.01,0.1,1,10} and repeat the experiment for 500 times. Which η results in the lowest average Eout over 500 experiments?

0.01

Question 14
Following Question 11, deepen your algorithm by making it capable of training a d-8-3-1 neural network with tanh-type neurons. Do not use any pre-training. Let r=0.1 and η=0.01 and repeat the experiment for 500 times. Which of the following is true about Eout over 500 experiments?

0.02≤Eout<0.04

Question 15
Experiment with 1 Nearest Neighbor
Implement any algorithm that `returns' the $1$ Nearest Neighbor hypothesis discussed in page 8 of lecture 214. gnbor(x)=ym such that x closest to xm
Run the algorithm on the following set for training:
hw4_knn_train.dat
and the following set for testing:
hw4_knn_test.dat

Which of the following is closest to Ein(gnbor)?

0.0

Question 16
Following Question 15, which of the following is closest to Eout(gnbor)?

0.34

Question 17
Now, implement any algorithm for the k Nearest Neighbor with k=5 to get g5-nbor(x). Run the algorithm on the same sets in Question 15 for training/testing.
Which of the following is closest to Ein(g5-nbor)?

0.2

Question 18
Following Question 17, Which of the following is closest to Eout(g5-nbor)

0.32

Question 19
Experiment with k-Means
Implement the $k$-Means algorithm (page 16 of lecture 214).Randomly select k instances from {xn} to initialize your μm Run the algorithm on the following set for training:
hw4_kmeans_train.dat
and repeat the experiment for 500 times. Calculate the clustering Ein by 1N∑Nn=1∑Mm=1[[xn∈Sm]]∥xn−μm∥2
as described on page 13 of lecture 214 for M=k.
For k=2, which of the following is closest to the average Ein of k-Means over 500 experiments?

2.5

Question 20
For k=10, which of the following is closest to the average Ein of k-Means over 500 experiments?

1.5
"""

import time
import numpy as np
from numpy import linalg as LA


def read_file(f):
    data = np.loadtxt(f)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x, y, len(data)


def init(shape, r):
    return [np.random.uniform(-r, r, (shape[i-1] + 1, shape[i])) for i in range(1, len(shape))]


def score(wl, x):
    x = np.append(1.0, x)
    return [x.dot(wl[:, i]) for i in range(len(wl[0]))]


def forward(x, w, shape):
    activation = []
    s_l = []
    for l in range(1, len(shape)):
        s = score(w[l - 1], x)
        s_l.append(s)
        x = np.tanh(score(w[l - 1], x))
        activation.append(x)
    return s_l, activation


def backward(y, w, s, activation):
    L = len(activation)
    delta = [-2 * (y - activation[-1]) * np.tanh(s[-1])]
    for l in range(L-2, -1, -1):
        if l == L - 2:
            delta = [np.array(delta[0][0] * w[l][1:, :]) * np.tanh(s[l])[0]] + delta
        else:
            pass
    return delta


def gradient_descent(x, w, activation, delta, eta):    # edited from http://www.cnblogs.com/xbf9xbf/p/4737525.html
    ret = []
    ret.append(w[0] - eta * np.array([np.hstack((1, x))]).transpose() * delta[0])
    for i in range(1, len(w), 1):
        ret.append(w[i] - eta * np.array([np.hstack((1, activation[i - 1]))]).transpose() * delta[i])
    return ret


def nnet(x, y, n):
    d = x.shape[1]
    M = 6
    r = 0.1
    shape = [d, M, 1]
    eta = 0.1
    w = np.array(init(shape, r))
    s, activation = forward(x[0], w, shape)
    # print(activation)
    delta = backward(y[0], w, s, activation)
    w = gradient_descent(x[0], w, activation, delta, eta)
    pass


def quiz11_14():
    x_train, y_train, n_train = read_file("hw4_nnet_train.dat")
    nnet(x_train, y_train, n_train)
    pass


def knn(x, y, k, xx):
    dis = np.sum((x - xx)**2, axis=1)
    order = np.argsort(dis)
    yy = 0
    for i in range(k):
        yy += y[order[i]]
    return np.sign(yy)


def knn_uniform(x, y, xx, gama):    # some thing wrong
    return y[np.argmax(np.exp(-gama * np.sum((x - xx) ** 2, axis=1)))]


def knn_e(x_tr, y_tr, k, x_ts, y_ts):
    yy = [knn(x_tr, y_tr, k, xx) for xx in x_ts]
    return np.count_nonzero(yy != y_ts) / len(y_ts)


def knn_uniform_e(x_tr, y_tr, gama, x_ts, y_ts):
    yy = [knn_uniform(x_tr, y_tr, xx, gama) for xx in x_ts]
    return np.count_nonzero(yy != y_ts) / len(y_ts)


def quiz15_18():
    x_train, y_train, n_train = read_file("hw4_knn_train.dat")
    x_test, y_test, n_test = read_file("hw4_knn_test.dat")
    for k in range(1, 1, 2):
        e_in = knn_e(x_train, y_train, k, x_train, y_train)
        print(k, ":  ", e_in)
        e_out = knn_e(x_train, y_train, k, x_test, y_test)
        print(k, ":  ", e_out)

    for gama in [0.001, 0.1, 1, 10, 100]:
        e_in = knn_uniform_e(x_train, y_train, gama, x_train, y_train)
        print(gama, ":  ", e_in)
        e_out = knn_uniform_e(x_train, y_train, gama, x_test, y_test)
        print(gama, ":  ", e_out)


def kmeans(x, n, k):
    order = list(range(n))
    np.random.shuffle(order)
    center = x[order[:k]]
    while 1:
        clusters = [[] for _ in range(k)]
        for xx in x:
            clusters[np.argmin(np.sum((center - xx)**2, axis=1))].append(xx)
        center_new = np.array([np.mean(clusters[i], axis=0) for i in range(k)])
        if LA.norm(center - center_new) == 0:
            return center, clusters
        center = center_new
    pass


def kmeans_e(n, k, center, clausters):
    dis = 0
    for kk in range(k):
        dis += np.sum([(center[kk] - c)**2 for c in clausters[kk]])
    return dis / n


def quiz19_20():
    x = np.loadtxt("hw4_kmeans_train.dat")
    n = len(x)
    T = 50
    for k in range(2, 12, 2):
        e_in = 0
        for t in range(T):
            center, clusters = kmeans(x, n, k)
            e_in += kmeans_e(n, k, center, clusters)
        e_in /= T
        print("k = ", k, "e_in = ", e_in)


def main():
    np.random.seed()
    start_time = time.time()
    # quiz11_14() # fall
    # quiz15_18()
    quiz19_20()
    print("\nTaken total %f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()

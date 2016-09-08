#!user/bin/env python3
# _*_ coding: utf-8 _*_


import time
import numpy as np
from numpy import linalg as la
from numpy import matlib as ml


def read_file(f):
    data = np.array([np.fromstring(line, dtype=float, sep=' ') for line in open(f, 'r').readlines()])
    x_d = data[:, :-1]
    y_d = data[:, -1].astype(int)
    return x_d, y_d, len(data)


def stump(s, i, theta, x):
    return s * np.sign(x[:, i] - theta).astype(int)


def err_g(x, y, s, i, theta, u):
    errs = (stump(s, i, theta, x) != y).astype(int)
    return np.average(errs, weights=u), errs


def G(x, g_stp):
    return np.sign(np.sum(np.array([stump(s, i, theta, x) * alpha for (s, i, theta, alpha) in g_stp]), 0)).astype(int)


def err_G(x, y, g_stp):
    return np.count_nonzero(G(x, g_stp) != y) / len(y)


def dsa(x, y, n, u):
    best_s = 1
    best_i = 0
    best_theta = 0
    best_errs = [0]*n
    e_in_min = n
    for i in range(len(x[0])):
        sort_idx = x[:, i].argsort()
        x_sorted = x[sort_idx]
        threshold = [x_sorted[0, i] - 0.01] + [(x_sorted[nn, i] + x_sorted[nn + 1, i]) / 2 for nn in range(n - 1)]
        for s in (-1, 1):
            for theta in threshold:
                e_in, errs = err_g(x, y, s, i, theta, u)
                if e_in < e_in_min:
                    best_s = s
                    best_i = i
                    best_theta = theta
                    best_errs = errs
                    e_in_min = e_in
    # print(best_s, best_i, best_theta, best_errs, e_in_min)
    return best_s, best_i, best_theta, best_errs


def adaboost(x, y, n):
    g_stp = []
    u = [1/n]*n
    T = 15
    for t in range(T):
        print(u)
        s, i, theta, errs = dsa(x, y, n, u)
        epsilon = np.average(errs, weights=u) / sum(u)
        scale = np.sqrt((1 - epsilon) / epsilon)
        for u_i in range(len(u)):
            if errs[u_i]:
                u[u_i] = u[u_i] * scale
            else:
                u[u_i] = u[u_i] / scale
        alpha = np.log(scale)
        g_stp.append((s, i, theta, alpha))
    return g_stp


def quiz12_18():
    X, Y, N = read_file("hw2_adaboost_train.dat")
    g_stp = adaboost(X, Y, N)
    e_in = err_G(X, Y, g_stp)
    X_t, Y_t, N_t = read_file("hw2_adaboost_test.dat")
    e_out = err_G(X_t, Y_t, g_stp)
    print(e_in, e_out)


# quiz19-20

def kernal(x, gamma, n):
    k = ml.empty((n, n))
    for i in range(n):
        for j in range(n):
            k[i, j] = np.exp(-gamma * la.norm(x[i, :] - x[j, :]) ** 2)
    return k


def KRG(x, y, gamma, lamb, n):
    return (lamb * np.eye(n) + kernal(x, gamma, n).I) * (np.matrix(y).T)


def err(x, y, model):
    (w, gamma, x_train) = model
    pre = np.zeros(len(y))
    for i in range(len(y)):
        for j in range(x_train.shape[0]):
            pre[i] = pre[i] + w[j] * np.exp(-1 * gamma * sum((x[i] - x_train[j]) ** 2))

    pre = np.sign(pre)
    return np.count_nonzero(pre != y) / len(y)

def quiz19_20():
    gamma_l = [32, 2, 0.125]
    lamb_l = [0.001, 1, 1000]
    data = np.loadtxt("hw2_lssvm_all.dat")
    x_train = data[:400, :-1]
    y_train = data[:400, -1].astype(int)
    x_test = data[400:, :-1]
    y_test = data[400:, -1].astype(int)
    n = len(y_train)
    print("gamma  lamb  e_in  e_out")
    for gamma in gamma_l:
        for lamb in lamb_l:
            w = np.array(KRG(x_train, y_train, gamma, lamb, n)).flatten()
            e_in = err(x_train, y_train, (w, gamma, x_train))
            e_out = err(x_test, y_test,  (w, gamma, x_train))
            print(gamma, "  ", lamb, "  ", e_in, "  ", e_out)


# quiz19-20


def main():
    np.random.seed()
    start_time = time.time()

    # quiz12_18()
    quiz19_20()

    print("\nTaken total %f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()

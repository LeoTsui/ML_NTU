#!user/bin/env python3
# _*_ coding: utf-8 _*_


import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def read_file(f):
    data = np.array([np.fromstring(line, dtype=float, sep=' ') for line in open(f, 'r').readlines()])
    x_d = data[:, 1:]
    y_d = data[:, 0].astype(int)
    return x_d, y_d, len(data)


def quiz15():
    X, Y, N = read_file("features.train")
    Y_0 = (Y == 0).astype(int)

    c_l = []
    w_l = []
    for i in range(-6, 4, 2):
        c = 10 ** i
        c_l.append(c)
        clf = svm.SVC(C=c, kernel='linear', shrinking=False)
        clf.fit(X, Y_0)
        w = clf.coef_.flatten()
        norm_w = np.linalg.norm(w, ord=2)
        w_l.append(norm_w)
        print("C = ", c, '    norm(w) =', norm_w)

    plt.semilogx(c_l, w_l)
    plt.savefig("h5_q15.png", dpi=300)


def quiz16():
    X, Y, N = read_file("features.train")
    for v in range(0, 10, 2):
        for i in range(-6, 4, 2):
            c = 10 ** i
            Y_v = (Y == v).astype(int)
            clf = svm.SVC(kernel='poly', C=c, degree=2, gamma=1, coef0=1)
            clf.fit(X, Y_v)
            Y_pre = clf.predict(X)
            e_in = np.count_nonzero(Y_pre != Y_v)
            print(v, "versus," "C = ", c, "e_in: ", e_in)


def quiz17():
    X, Y, N = read_file("features.train")
    for v in range(0, 10, 2):
        for i in range(-6, 4, 2):
            c = 10 ** i
            Y_v = (Y == v).astype(int)
            clf = svm.SVC(kernel='poly', C=c, degree=2, gamma=1, coef0=1)
            clf.fit(X, Y_v)
            Y_pre = clf.predict(X)
            e_in = np.count_nonzero(Y_pre != Y_v)
            sum_alpha = np.sum(np.abs(clf.dual_coef_))
            print(v, "versus," "C=", c, "    e_in: ", e_in, "    sum_alpha: ", sum_alpha)
        print('\n')


def quiz19():
    X, Y, N = read_file("features.train")
    X_out, Y_out, N_out = read_file("features.test")
    Y_0 = (Y == 0).astype(int)
    Y_out_0 = (Y_out == 0).astype(int)
    for g in range(0, 5):
        gam = 10**g
        clf = svm.SVC(kernel='rbf', C=0.1, gamma=gam)
        clf.fit(X, Y_0)
        Y_pre = clf.predict(X_out)
        e_out = np.count_nonzero(Y_pre != Y_out_0)
        print("C=", 0.1, "gamma=", gam,  " e_out: ", e_out)


def quiz20():
    X, Y, N = read_file("features.train")
    Y_0 = (Y == 0).astype(int)

    gamma_count = [0]*5

    for t in range(100):
        print("times: ", t)
        idx = np.arange(N)
        np.random.shuffle(idx)
        x_val = X[idx[:1000]]
        y_val = Y_0[idx[:1000]]
        x_val_test = X[idx[1000:]]
        y_val_test = Y_0[idx[1000:]]
        e_val_count = [0]*5

        for g in range(0, 5):
            gam = 10 ** g
            clf = svm.SVC(kernel='rbf', C=0.1, gamma=gam)
            clf.fit(x_val, y_val)
            Y_pre = clf.predict(x_val_test)
            e_val = np.count_nonzero(Y_pre != y_val_test)
            e_val_count[g] = e_val

        print(e_val_count)
        gamma_count[np.argmin(e_val_count)] += 1

    print(gamma_count, 10**np.argmax(gamma_count))

def main():
    np.random.seed()
    start_time = time.time()

    # quiz15()
    # quiz16()
    # quiz17()
    # quiz19()
    quiz20()

    print("\nTaken total %f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()

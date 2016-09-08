#!user/bin/env python3
# _*_ coding: utf-8 _*_

"""
Question 13
Experiment with Regularized Linear Regression and Validation

Consider regularized linear regression (also called ridge regression) for classification.
wreg=argminwλN∥w∥2+1N∥Xw−y∥2,
Run the algorithm on the following data set as D

https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw4%2Fhw4_train.dat

and the following set for evaulating Eout

https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw4%2Fhw4_test.dat

Because the data sets are for classification, please consider only the 0/1 error for all the problems below.
Let λ=10, which of the followings is the corresponding Ein and Eout?

Ein=0.050, Eout=0.045

Question 14
Among log10λ={2,1,0,−1,…,−8,−9,−10}. What is the λ with the minimum Ein? Compute λ and its corresponding Ein and Eout then select the closest answer. Break the tie by selecting the largest λ

log10λ=−8,Ein=0.015,Eout=0.020

Question 15
Among log10λ={2,1,0,−1,…,−8,−9,−10}. What is the λ with the minimum Eout? Compute λ and the corresponding Ein and Eout then select the closest answer. Break the tie by selecting the largest λ.

log10λ=−7,Ein=0.030,Eout=0.015

Question 16
Now split the given training examples in D to the first 120 examples for Dtrain and 80 for Dval.

Ideally, you should randomly do the 120/80 split. Because the given examples are already randomly permuted, however, we would use a fixed split for the purpose of this problem.

Run the algorithm on Dtrain to get g−λ, and validate g−λ with Dval.
Among log10λ={2,1,0,−1,…,−8,−9,−10}. What is the λ with the minimum Etrain(g−λ)? Compute λ and the corresponding Etrain(g−λ), Eval(g−λ) and Eout(g−λ) then select the closet answer. Break the tie by selecting the largest λ.

log10λ=−8,Etrain(g−λ)=0.000,Eval(g−λ)=0.050,Eout(g−λ)=0.025

Question 17
Among log10λ={2,1,0,−1,…,−8,−9,−10}. What is the λ with the minimum Eval(g−λ)? Compute λ and the corresponding Etrain(g−λ), Eval(g−λ) and Eout(g−λ) then select the closet answer. Break the tie by selecting the largest λ.

log10λ=0,Etrain(g−λ)=0.033,Eval(g−λ)=0.038,Eout(g−λ)=0.028

Question 18
Run the algorithm with the optimal λ of the previous problem on the whole D to get gλ. Compute Ein(gλ) and Eout(gλ) then select the closet answer.

Ein(gλ)=0.035,Eout(gλ)=0.020

Question 19
Now split the given training examples in D to five folds, the first 40 being fold 1, the next 40 being fold 2, and so on. Again, we take a fixed split because the given examples are already randomly permuted.

Among log10λ={2,1,0,−1,…,−8,−9,−10}. What is the λ with the minimum Ecv, where Ecv comes from the five folds defined above? Compute λ and the corresponding Ecv then select the closet answer. Break the tie by selecting the largest λ.

log10λ=−8,Ecv=0.030

Question 20
Run the algorithm with the optimal λ of the previous problem on the whole D to get gλ. Compute Ein(gλ) and Eout(gλ) then select the closet answer.

Ein(gλ)=0.015,Eout(gλ)=0.020
"""

import time
import numpy as np


def ridge_reg(x, y, lmd):
    z = np.linalg.inv(np.dot(x.transpose(), x) + lmd * np.eye(x.shape[1]))
    return np.dot(np.dot(z, x.transpose()), y)


def err_01(x, y, w):
    return np.sign(np.dot(x, w)) != y


def err_func(x, y, w, n):
    e = 0
    for i in range(n):
        if err_01(x[i], y[i], w):
            e += 1
    return e / n


def read_file(f):
    x_d = []
    y_d = []
    with open(f, 'r') as d:
        for line in d:
            l = line.split()
            x = [1.0] + [float(v) for v in l[: -1]]
            x_d.append(x)
            y_d.append(int(l[-1]))
    return np.array(x_d), np.array(y_d), len(y_d)


def quiz13(lmd=10):
    x_in, y_in, n_in = read_file("hw4_train.dat")
    x_out, y_out, n_out = read_file("hw4_test.dat")
    w_reg = np.array(ridge_reg(x_in, y_in, lmd)).flatten()
    e_in = err_func(x_in, y_in, w_reg, n_in)
    e_out = err_func(x_out, y_out, w_reg, n_out)
    return e_in, e_out


def quiz14():
    x_in, y_in, n_in = read_file("hw4_train.dat")
    x_out, y_out, n_out = read_file("hw4_test.dat")
    best_e_in = float("inf")
    best_lmd = 0
    w = 0
    for lmd in range(2, -11, -1):
        w_reg = np.array(ridge_reg(x_in, y_in, pow(10, lmd))).flatten()
        e_in = err_func(x_in, y_in, w_reg, n_in)
        if e_in < best_e_in:
            best_e_in = e_in
            w = w_reg
            best_lmd = lmd
    e_out = err_func(x_out, y_out, w, n_out)
    return best_lmd, best_e_in, e_out


def quiz15():
    x_in, y_in, n_in = read_file("hw4_train.dat")
    x_out, y_out, n_out = read_file("hw4_test.dat")
    best_e_out = float("inf")
    best_lmd = 0
    w = 0
    for lmd in range(2, -11, -1):
        w_reg = np.array(ridge_reg(x_in, y_in, pow(10, lmd))).flatten()
        e_out = err_func(x_out, y_out, w_reg, n_out)
        if e_out < best_e_out:
            best_e_out = e_out
            w = w_reg
            best_lmd = lmd
    e_in = err_func(x_in, y_in, w, n_in)
    return best_lmd, e_in, best_e_out


def quiz16():
    x_in, y_in, n_in = read_file("hw4_train.dat")
    x_out, y_out, n_out = read_file("hw4_test.dat")
    n_train = 120
    n_val = 80
    x_train = x_in[:120]
    y_train = y_in[:120]
    x_val = x_in[120:]
    y_val = y_in[120:]
    best_e_train = float("inf")
    best_lmd = 0
    w = 0
    for lmd in range(2, -11, -1):
        w_reg = np.array(ridge_reg(x_train, y_train, pow(10, lmd))).flatten()
        e_train = err_func(x_train, y_train, w_reg, n_train)
        if e_train < best_e_train:
            best_e_train = e_train
            w = w_reg
            best_lmd = lmd
    e_out = err_func(x_out, y_out, w, n_out)
    e_val = err_func(x_val, y_val, w, n_val)
    return best_lmd, best_e_train, e_val, e_out


def quiz17():
    x_in, y_in, n_in = read_file("hw4_train.dat")
    x_out, y_out, n_out = read_file("hw4_test.dat")
    n_train = 120
    n_val = 80
    x_train = x_in[:n_train]
    y_train = y_in[:n_train]
    x_val = x_in[120:]
    y_val = y_in[120:]
    best_e_val = float("inf")
    best_lmd = 0
    w = 0
    for lmd in range(2, -11, -1):
        w_reg = np.array(ridge_reg(x_train, y_train, pow(10, lmd))).flatten()
        e_val = err_func(x_val, y_val, w_reg, n_val)
        if e_val < best_e_val:
            best_e_val = e_val
            w = w_reg
            best_lmd = lmd
    e_train = err_func(x_train, y_train, w, n_train)
    e_out = err_func(x_out, y_out, w, n_out)
    return best_lmd, e_train, best_e_val, e_out


def quiz18():
    x_in, y_in, n_in = read_file("hw4_train.dat")
    x_out, y_out, n_out = read_file("hw4_test.dat")
    n_train = 120
    n_val = 80
    x_train = x_in[:n_train]
    y_train = y_in[:n_train]
    x_val = x_in[120:]
    y_val = y_in[120:]
    best_e_val = float("inf")
    best_lmd = 0
    for lmd in range(2, -11, -1):
        w_reg = np.array(ridge_reg(x_train, y_train, pow(10, lmd))).flatten()
        e_val = err_func(x_val, y_val, w_reg, n_val)
        if e_val < best_e_val:
            best_e_val = e_val
            best_lmd = lmd
    return quiz13(pow(10, best_lmd))


def quiz1920(split=40):
    x_in, y_in, n_in = read_file("hw4_train.dat")
    x_out, y_out, n_out = read_file("hw4_test.dat")
    n_cv = split
    best_e_cv = float("inf")
    best_lmd = 0
    for lmd in range(2, -11, -1):
        e_cv = 0
        for i in range(int(n_in / n_cv)):
            x_cv = x_in[n_cv * i: n_cv * (i + 1)]
            y_cv = y_in[n_cv * i: n_cv * (i + 1)]
            w_reg = np.array(ridge_reg(x_cv, y_cv, pow(10, lmd))).flatten()
            e_cv += err_func(x_cv, y_cv, w_reg, n_cv)
        print(lmd, e_cv)
        if e_cv < best_e_cv:
            best_e_cv = e_cv
            best_lmd = lmd
    w = np.array(ridge_reg(x_in, y_in, pow(10, best_lmd))).flatten()
    e_in = err_func(x_in, y_in, w, n_in)
    e_out = err_func(x_out, y_out, w, n_out)
    return best_lmd, best_e_cv, e_in, e_out



def main():
    np.random.seed()
    start_time = time.time()
    # print("q13: \n", quiz13())
    # print("q14: \n", quiz14())
    # print("q15: \n", quiz15())
    # print("q16: \n", quiz16())
    # print("q17: \n", quiz17())
    # print("q18: \n", quiz18())
    print("q19: \n", quiz1920())
    print("Taken total %f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()

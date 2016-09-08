#!user/bin/env python3
# _*_ coding: utf-8 _*_

"""
Question 13
For Questions 13-15, you will play with linear regression and feature transforms. Consider the target function:
f(x1,x2)=sign(x21+x22−0.6)
Generate a training set of N=1000 points on X=[−1,1]×[−1,1] with uniform probability of picking each x∈X. Generate simulated noise by flipping the sign of the output in a random 10% subset of the generated training set.
Carry out Linear Regression without transformation, i.e., with feature vector: (1,x1,x2), to find the weight w, and use wlin directly for classification. What is the closest value to the classification (0/1) in-sample error (Ein)? Run the experiment 1000 times and take the average Ein in order to reduce variation in your results.

0.5

Question 14
Now, transform the training data into the following nonlinear feature vector:
(1,x1,x2,x1x2,x21,x22)
Find the vector w~ that corresponds to the solution of Linear Regression, and take it for classification.
Which of the following hypotheses is closest to the one you find using Linear Regression on the transformed input? Closest here means agrees the most with your hypothesis (has the most probability of agreeing on a randomly selected point).

g(x1,x2)=sign(−1−0.05x1+0.08x2+0.13x1x2+1.5x21+1.5x22)

Question 15
What is the closest value to the classification out-of-sample error Eout of your hypothesis? Estimate it by generating a new set of 1000 points and adding noise as before. Average over 1000 runs to reduce the variation in your results.

0.1

Question 18
For Questions 18-20, you will play with logistic regression.

Please use the following set for training:

https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw3%2Fhw3_train.dat

and the following set for testing:

https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw3%2Fhw3_test.dat

Implement the fixed learning rate gradient descent algorithm for logistic regression. Run the algorithm with η=0.001 and T=2000, what is Eout(g) from your algorithm, evaluated using the 0/1 error on the test set?

0.475

Question 19
Implement the fixed learning rate gradient descent algorithm for logistic regression. Run the algorithm with η=0.01 and T=2000, what is Eout(g) from your algorithm, evaluated using the 0/1 error on the test set?

0.220

Question 20
Implement the fixed learning rate stochastic gradient descent algorithm for logistic regression. Instead of randomly choosing n in each iteration, please simply pick the example with the cyclic order n=1,2,…,N,1,2,….
Run the algorithm with η=0.001 and T=2000, what is Eout(g) from your algorithm, evaluated using the 0/1 error on the test set?

0.475

"""

import time
import numpy as np


def target_func(x1, x2):
    return int(np.sign(x1**2 + x2**2 - 0.6))


def gen_data(a=-1.0, b=1.0, size=1000):
    x0 = [1]*size
    x1 = (b - a) * np.random.random(size) + a
    x2 = (b - a) * np.random.random(size) + a
    x = np.column_stack((x0, x1, x2))
    # x = np.column_stack((x0, x1, x2, x1 * x2, x1 * x1, x2 * x2))
    y = [target_func(x1[i], x2[i]) for i in range(size)] * np.where(np.random.random(size) < 0.2, -1, 1)
    return x, y


def get_wlin(x, y):
    wlin = np.linalg.pinv(x) * np.matrix(y).getT()
    return wlin


def lin_reg_e_in(x, y, wlin):
    e_in = 0
    for i in range(len(x)):
        if np.sign(x[i] * wlin) != y[i]:
            e_in += 1
    return e_in / len(x)


def lin_reg_e_out(wlin, size):
    x, y = gen_data(-1.0, 1.0, size)
    e_out = lin_reg_e_in(x, y, wlin)
    return e_out


def quiz1315(size=1000, the_iter=1000):
    e_in_count = 0
    e_out_count = 0
    for i in range(the_iter):
        x, y = gen_data(-1.0, 1.0, size)
        wlin = get_wlin(x, y)
        # wlin = np.matrix([[-1], [-0.05], [0.08], [0.13], [1.5], [1.5]])
        e_in_count += lin_reg_e_in(x, y, wlin)
        e_out_count += lin_reg_e_out(wlin, size)
    avg_e_in = e_in_count / the_iter
    avg_e_out = e_out_count / the_iter
    return wlin, avg_e_in, avg_e_out


def read_file(f):
    x_d = []
    y_d = []
    with open(f, 'r') as d:
        for line in d:
            l = line.split()
            x = [float(v) for v in l[: -1]]
            x_d.append(x)
            y_d.append(int(l[-1]))
    return np.array(x_d), np.array(y_d)


def logic_func(s):
    return float(1 / (1 + np.exp(-s)))


def grad_func(x, y, w):
    # x[N*d] y[1*N] w[1*d]
    d = len(w)
    n = len(y)
    grad_mat = np.matrix([0.0] * d)
    for i in range(n):
        grad_mat += logic_func(-y[i] * np.dot(w, x[i])) * (-y[i] * x[i])
    return grad_mat / n


def grad_descent(x, y, eta, t):
    d = len(x[0])
    w = np.array([1.0] * d)
    for i in range(t):
        w = w - eta * np.array(grad_func(x, y, w)).flatten()
    return w


def sgd(x, y, eta, t):
    d = len(x[0])
    w = np.array([1.0] * d)
    for i in range(t):
        find_err = False
        while not find_err:
            n = np.random.randint(0, len(y))
            if np.sign(np.dot(w, x[n])) != y[n]:
                w = w - eta * np.array(logic_func(-y[n] * np.dot(w, x[n])) * (y[n] * x[n])).flatten()
                find_err = True
    return w


def logic_reg_e_out(x, y, w):
    e_out = 0
    for i in range(len(y)):
        if np.sign(np.dot(w, x[i])) != y[i]:
            e_out += 1
    return float(e_out) / float(len(y))


def quiz1820(eta, t):
    x, y = read_file("hw3_train.dat")
    # w = grad_descent(x, y, eta, t)
    w = sgd(x, y, eta, t)
    x, y = read_file("hw3_test.dat")
    e_out = logic_reg_e_out(x, y, w)
    return w, e_out


def main():
    np.random.seed()
    start_time = time.time()
    # print(quiz1315(1000, 1000))
    # print(quiz1820(0.001, 2000))
    # print(quiz1820(0.01, 2000))
    print(quiz1820(0.001, 2000))
    print("Taken total %f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()

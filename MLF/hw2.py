#!user/bin/env python3
# _*_ coding: utf-8 _*_

"""
Question 16
For Questions 16-20, you will play with the decision stump algorithm.

In class, we taught about the learning model of ``positive and negative rays'' (which is simply one-dimensional perceptron) for one-dimensional data. The model contains hypotheses of the form:
hs,θ(x)=s⋅sign(x−θ).
The model is frequently named the ``decision stump'' model and is one of the simplest learning models. As shown in class, for one-dimensional data, the VC dimension of the decision stump model is 2.

In fact, the decision stump model is one of the few models that we could easily minimize Ein efficiently by enumerating all possible thresholds. In particular, for N examples, there are at most 2N dichotomies (see page 22 of class05 slides), and thus at most 2N different Ein values. We can then easily choose the dichotomy that leads to the lowest Ein, where ties can be broken by randomly choosing among the lowest-Ein ones. The chosen dichotomy stands for a combination of some `spot' (range of θ) and s, and commonly the median of the range is chosen as the θ that realizes the dichotomy.

In this problem, you are asked to implement such and algorithm and run your program on an artificial data set. First of all, start by generating a one-dimensional data by the procedure below:

(a) Generate x by a uniform distribution in [−1,1].
[b) Generate y by f(x)=s~(x) + noise where s~(x)=sign(x) and the noise flips the result with 20% probability.

For any decision stump hs,θ with θ∈[−1,1], express Eout(hs,θ) as a function of θ and s.

0.5+0.3s(|θ|−1)

no noise Eout(hs,θ) = 0.5+0.5s

Question 17
Generate a data set of size 20 by the procedure above and run the one-dimensional decision stump algorithm on the data set. Record Ein and compute Eout with the formula above. Repeat the experiment (including data generation, running the decision stump algorithm, and computing Ein and Eout) 5,000 times. What is the average Ein? Choose the closest option.

0.15

Question 18
Continuing from the previous question, what is the average Eout? Choose the closest option.

0.25

Question 19
Decision stumps can also work for multi-dimensional data. In particular, each decision stump now deals with a specific dimension i, as shown below.
hs,i,θ(x)=s⋅sign(xi−θ).
Implement the following decision stump algorithm for multi-dimensional data:

a) for each dimension i=1,2,⋯,d, find the best decision stump hs,i,θ using the one-dimensional decision stump algorithm that you have just implemented.
b) return the ``best of best'' decision stump in terms of Ein. If there is a tie, please randomly choose among the lowest-Ein ones.

The training data Dtrain is available at:

https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw2%2Fhw2_train.dat

The testing data Dtest is available at:

https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw2%2Fhw2_test.dat

Run the algorithm on the Dtrain. Report the Ein of the optimal decision stump returned by your program. Choose the closest option.

0.25

Question 20
Use the returned decision stump to predict the label of each example within the Dtest. Report an estimate of Eout by Etest. Choose the closest option.

0.35
"""

import math
import numpy as np


def hypo(x, s, theta):
    return s * np.sign(x - theta)


def decision_stump(x_data, y_data):
    point_list = [float(math.ceil(x_data[0]))] + [float(x) for x in x_data] + [float(math.floor(x_data[-1]))]
    theta_list = [(point_list[i] + point_list[i + 1]) / 2 for i in range(len(point_list) - 1)]
    best_e_in = float("inf")
    best_s = 0
    best_theta = 0
    for theta in theta_list:
        for s in [-1, 1]:
            e_in = decision_stump_e_in(x_data, y_data, s, theta)
            if e_in < best_e_in:
                best_e_in = e_in
                best_s = s
                best_theta = theta
    return best_s, best_theta, best_e_in


def decision_stump_e_in(x_data, y_data, s, theta):
    e_in = 0
    for i in range(len(x_data)):
        if hypo(x_data[i], s, theta) != y_data[i]:
            e_in += 1
    return e_in / len(x_data)


def decision_stump_e_out(s, theta):
    return 0.5 + 0.3 * s * (abs(theta) - 1)


def e_out_test(s, theta, dim, test):
    data = load_data(test)
    dimension = len(data[0]) - 1

    dat = np.array(sorted(data, key=lambda x: x[dim]))
    x_d = dat[:, dim]
    y_d = dat[:, dimension]
    e_out = decision_stump_e_in(x_d, y_d, s, theta)

    return e_out


def gen_data(a=-1.0, b=1.0, size=20):
    x_data = np.sort((b - a) * np.random.random(size) + a)
    y_data = np.sign(x_data) * np.where(np.random.random(size) < 0.2, -1, 1)
    return x_data, y_data


def load_data(f):
    data = []
    with open(f, 'r') as d:
        for line in d:
            l = line.split()
            d = [float(v) for v in l]
            data.append(d)
    return np.array(data)


def quiz1718(the_iter=5000):
    e_in_count = 0
    e_out_count = 0
    for i in range(the_iter):
        x_d, y_d = gen_data()
        s, theta, e_in = decision_stump(x_d, y_d)
        e_in_count += e_in
        e_out_count += decision_stump_e_out(s, theta)
    arg_e_in = e_in_count / the_iter
    arg_e_out = e_out_count / the_iter
    return arg_e_in, arg_e_out


def quiz1920(train, test):
    data = load_data(train)
    dimension = len(data[0]) - 1

    ret_l = []    # (s, theta, e_in)

    for i in range(dimension):
        dat = np.array(sorted(data, key=lambda x: x[i]))
        x_d = dat[:, i]
        y_d = dat[:, dimension]
        ret_l.append(decision_stump(x_d, y_d))
    dim = ret_l.index(min(ret_l, key=lambda r: r[2]))
    s, theta, e_in = ret_l[dim]

    e_out = e_out_test(s, theta, dim, test)

    return s, theta, dim, e_in, e_out


def main():
    train_data = "hw2_train.dat"
    test_data = "hw2_test.dat"
    np.random.seed()

    print(quiz1718(5000))
    print(quiz1920(train_data, test_data))

if __name__ == "__main__":
    main()

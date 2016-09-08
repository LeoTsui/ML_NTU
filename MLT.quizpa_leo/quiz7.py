#!user/bin/env python3
# _*_ coding: utf-8 _*_

"""
Question 13
Experiments with Decision Tree
Implement the simple C&RT algorithm without pruning using the Gini index as the impurity measure as introduced in the class. For the decision stump used in branching, if you are branching with feature i and direction s, please sort all the xn,i values to form (at most) N+1 segments of equivalent θ, and then pick θ within the median of the segment. Run the algorithm on the following set for training:
hw3_train.dat
and the following set for testing:
hw3_test.dat
How many internal nodes (branching functions) are there in the resulting tree G?
10

Question 14
Continuing from Question 13, which of the following is closest to the Ein (evaluated with 0/1 error) of the tree?
0.0

Question 15
Continuing from Question 13, which of the following is closest to the Eout (evaluated with 0/1 error) of the tree?
0.15

Question 16
Now implement the Bagging algorithm with N′=N and couple it with your decision tree above to make a preliminary random forest GRS. Produce T=300 trees with bagging. Repeat the experiment for 100 times and compute average Ein and Eout using the 0/1 error.
Which of the following is true about the average Ein(gt) for all the 30000 trees that you have generated?
0.03≤average Ein(gt)<0.06

Question 17
Continuing from Question 16, which of the following is true about the average Ein(GRF)?
0.00≤average Ein(GRF)<0.03

Question 18
Continuing from Question 16, which of the following is true about the average Eout(GRF)?
0.06≤average Eout(GRF)<0.09

Question 19
Now, `prune' your decision tree algorithm by restricting it to have one branch only. That is, the tree is simply a decision stump determined by Gini index. Make a random `forest' GRS with those decision stumps with Bagging like Questions 16-18 with T=300. Repeat the experiment for 100 times and compute average Ein and Eout using the 0/1 error.
Which of the following is true about the average Ein(GRS)?
0.09≤average Ein(GRS)<0.12

Question 20
Continuing from Question 19, which of the following is true about the average Eout(GRS)?
0.12≤average Eout(GRS)<0.50
"""


import time
import numpy as np


class decision_tree_node:
    def __init__(self, _s, _i, _theta, l=True):
        self.s = _s
        self.i = _i
        self.theta = _theta
        self.is_leaf = l
        self.sign = 0
        self.l = None
        self.r = None


def read_file(f):
    data = np.loadtxt(f)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x, y, len(data)


def stump(s, i, theta, x):
    return s * np.sign(x[:, i] - theta).astype(int)


def stump_x(s, i, theta, x):
    return s * np.sign(x[i] - theta).astype(int)


def err_01(y_pre, y):    # 1, -1
    return np.count_nonzero(y_pre != y) / len(y)


def gini_index(y, n):
    return (1 - ((sum(y == 1) / n) ** 2) - ((sum(y == -1) / n) ** 2)) if n else 0


def branch_array(s, i, theta, x, y, n):
    x_l = np.empty(shape=[0, 2])
    y_l = []
    x_r = np.empty(shape=[0, 2])
    y_r = []
    y_pre = stump(s, i, theta, x)
    for nn in range(n):
        if y_pre[nn] == 1:
            x_r = np.append(x_r, [x[nn, :]], axis=0)
            y_r = np.append(y_r, y[nn]).astype(int)
        else:
            x_l = np.append(x_l, [x[nn, :]], axis=0)
            y_l = np.append(y_l, y[nn]).astype(int)
    return x_l, y_l, x_r, y_r


def impurity(s, i, theta, x, y, n):
    x_l, y_l, x_r, y_r = branch_array(s, i, theta, x, y, n)
    return len(y_l) * gini_index(y_l, len(y_l)) + len(y_r) * gini_index(y_r, len(y_r))


def decision_stump_alg(x, y, n):
    best_s = 0
    best_i = 0
    best_theta = 0
    b_min = n
    if n > 0:
        for i in range(len(x[0])):
            sort_idx = x[:, i].argsort()
            x_sorted = x[sort_idx]
            threshold = [x_sorted[0, i] - 0.01] + [(x_sorted[nn, i] + x_sorted[nn + 1, i]) / 2 for nn in range(n - 1)] + [x_sorted[-1, i] + 0.01]
            for s in (-1, 1):
                for theta in threshold:
                    b = impurity(s, i, theta, x, y, n)
                    if b < b_min:
                        best_s = s
                        best_i = i
                        best_theta = theta
                        b_min = b
    return best_s, best_i, best_theta


def decision_tree(x, y, n):
    s, i, theta = decision_stump_alg(x, y, n)    # s sign -1, 1. 0 means terminate
    
    if (np.unique(x).size == 1) or (np.unique(y).size == 1):
        node = decision_tree_node(0, 0, 0)
        node.sign = y[0]
        return node
    elif s != 0:
        node = decision_tree_node(s, i, theta, False)
        x_l, y_l, x_r, y_r = branch_array(s, i, theta, x, y, n)
        node.l = decision_tree(x_l, y_l, len(y_l))
        node.r = decision_tree(x_r, y_r, len(y_r))
        return node
    else:
        return None


def random_forest(x, y, n, T):
    forest = []
    for t in range(T):
        picked_idx = np.random.randint(n, size=n)
        root = decision_tree(x[picked_idx], y[picked_idx], n)
        forest.append(root)
        # print("e_in: ", err_01(predict(x, root), y))    # Ein(gt)
    return forest


def predict_x(x, node):
    s = node.s
    i = node.i
    theta = node.theta
    if node.is_leaf:
        return node.sign
    elif stump_x(s, i, theta, x) == -1:
        return predict_x(x, node.l)
    else:
        return predict_x(x, node.r)


def predict(x, node):
    return [predict_x(xx, node) for xx in x]


def predict_forest(x, forest):
    y_pre = [predict(x, tree) for tree in forest]
    return np.sign(np.sum(y_pre, 0)).astype(int)


def random_stump_forest(x, y, n, T):
    forest = []
    for t in range(T):
        picked_idx = np.random.randint(n, size=n)
        s, i, theta = decision_stump_alg(x[picked_idx], y[picked_idx], n)
        forest.append((s, i, theta))
    return forest


def predict_stump_forest(x, forest):
    y_pre = [stump(s, i, theta, x) for (s, i, theta) in forest]
    return np.sign(np.sum(y_pre, 0)).astype(int)


def quiz13_15():
    x, y, n = read_file("hw3_train.dat")
    x_test, y_test, n_test = read_file("hw3_test.dat")
    root = decision_tree(x, y, n)
    print("e_in: ", err_01(predict(x, root), y))
    print("e_out: ", err_01(predict(x_test, root), y_test))


def quiz16_18(T=10, times=10):
    x, y, n = read_file("hw3_train.dat")
    x_test, y_test, n_test = read_file("hw3_test.dat")
    e_in = 0
    e_out = 0
    for i in range(times):
        forest = random_forest(x, y, n, T)
        e_in += err_01(predict_forest(x, forest), y)
        e_out += err_01(predict_forest(x_test, forest), y_test)

    e_in /= times
    e_out /= times
    print(e_in, e_out)


def quiz19_20(T=3, times=10):
    x, y, n = read_file("hw3_train.dat")
    x_test, y_test, n_test = read_file("hw3_test.dat")
    e_in = 0
    e_out = 0
    for i in range(times):
        forest = random_stump_forest(x, y, n, T)
        e_in += err_01(predict_stump_forest(x, forest), y)    # err!!
        e_out += err_01(predict_stump_forest(x_test, forest), y_test)

    e_in /= times
    e_out /= times
    print(e_in, e_out)


def main():
    np.random.seed()
    start_time = time.time()
    # quiz13_15()
    # quiz16_18()
    quiz19_20()
    print("\nTaken total %f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()

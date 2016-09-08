#!user/bin/env python3
# _*_ coding: utf-8 _*_

"""
Question 15
For Questions 15-20, you will play with PLA and pocket algorithm. First, we use an artificial data set to study PLA. The data set is in
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat
Each line of the data set contains one (xn,yn) with xn∈R4. The first 4 numbers of the line contains the components of xn orderly, the last number is yn.
Please initialize your algorithm with w=0 and take sign(0) as −1
Implement a version of PLA by visiting examples in the naive cycle using the order of examples in the data set. Run the algorithm on the data set. What is the number of updates before the algorithm halts?

31 - 50 updates

Question 16
Implement a version of PLA by visiting examples in fixed, pre-determined random cycles throughout the algorithm. Run the algorithm on the data set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average number of updates before the algorithm halts?

31 - 50 updates

Question 17
Implement a version of PLA by visiting examples in fixed, pre-determined random cycles throughout the algorithm, while changing the update rule to be
wt+1←wt+ηyn(t)xn(t)
with η=0.5. Note that your PLA in the previous Question corresponds to η=1. Please repeat your experiment for 2000 times, each with a different random seed. What is the average number of updates before the algorithm halts?

31 - 50 updates

Question 18
Next, we play with the pocket algorithm. Modify your PLA in Question 16 to visit examples purely randomly, and then add the `pocket' steps to the algorithm. We will use
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_train.dat
as the training data set D, and
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_test.dat
as the test set for ``verifying'' the g returned by your algorithm (see lecture 4 about verifying). The sets are of the same format as the previous one.
Run the pocket algorithm with a total of 50 updates on D, and verify the performance of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?

<0.2

Question 19
Modify your algorithm in Question 18 to return w50 (the PLA vector after 50 updates) instead of w^ (the pocket vector) after 50 updates. Run the modified algorithm on D, and verify the performance using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?

0.2 - 0.4

Question 20
Modify your algorithm in Question 18 to run for 100 updates instead of 50, and verify the performance of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?

<0.2
"""

import numpy as np
import random


def read_file(f):
    x_d = []
    y_d = []
    with open(f, 'r') as d:
        for line in d:
            l = line.split()
            x = [1.0] + [float(v) for v in l[: -1]]
            x_d.append(x)
            y_d.append(int(l[-1]))
    return np.array(x_d), np.array(y_d)


def native_pla(x_d, y_d, is_rand=False, repeat=1, eta=1.0):
    total_update = 0

    for rpt in range(0, repeat):
        w = np.zeros(len(x_d[0]))
        update_count = 0
        all_pass = False

        index = [i for i in range(len(x_d))]
        if is_rand:
            random.shuffle(index)

        while not all_pass:
            all_pass = True
            for t in index:
                if np.sign(np.inner(x_d[t], w)) != y_d[t]:
                    w += eta * y_d[t] * x_d[t]
                    all_pass = False
                    update_count += 1

        total_update += update_count

    return w, total_update / repeat


def pocket_pla(x_d, y_d, x_t, y_t, update, repeat=1, eta=1.0):
    error = 0
    random.seed()
    
    for rpt in range(0, repeat):
        w = np.zeros(len(x_d[0]))
        wg = w
        err_wg = test_pocket_pla(x_d, y_d, wg)

        # index = [i for i in range(len(x_d))]
        # random.shuffle(index)

        for i in range(update):
            #for t in index:
            find_err = False
            while not find_err:
                t = random.randint(0, (len(x_d) - 1))
                if np.sign(np.inner(x_d[t], w)) != y_d[t]:
                    w += eta * y_d[t] * x_d[t]
                    find_err = True

            err_w = test_pocket_pla(x_d, y_d, w)
            if err_w < err_wg:
                wg = w
                err_wg = err_w

        error += test_pocket_pla(x_t, y_t, wg)
        # Q 19
        # error += test_pocket_pla(x_t, y_t, w)

    return wg, error / repeat


def test_pocket_pla(x_t, y_t, w):
    err = 0
    for i in range(len(x_t)):
        if np.sign(np.inner(x_t[i], w)) != y_t[i]:
            err += 1
    return err / len(x_t)


def main():

    # Q 15-17
    data15 = "hw1_15_train.dat"
    x_data, y_data = read_file(data15)

    # Q 15, native PLA
    print("Q 15: ", native_pla(x_data, y_data)[1])

    # Q 16, fixed, pre-determined random cycles
    # print("Q 16: ", native_pla(x_data, y_data, True, 2000)[1])

    # Q 17, fixed, pre-determined random cycles, with η=0.5
    # print("Q 17: ", native_pla(x_data, y_data, True, 2000, 0.5)[1])

    # Q 18-20
    data18_train = "hw1_18_train.dat"
    data18_test = "hw1_18_test.dat"
    x_data, y_data = read_file(data18_train)
    x_test, y_test = read_file(data18_test)

    # Q 18, purely randomly, total 50 updates, repeat 2000
    print("Q 18: ", pocket_pla(x_data, y_data, x_test, y_test, 50, 100)[1])
    # Q 19, purely randomly, total 50 updates, repeat 2000, w = w50
    # print("Q 19: ", pocket_pla(x_data, y_data, x_test, y_test, 50, 100)[1])
    # Q 20, purely randomly, total 100 updates, repeat 2000
    # print("Q 20: ", pocket_pla(x_data, y_data, x_test, y_test, 100, 100)[1])


if __name__ == "__main__":
    main()

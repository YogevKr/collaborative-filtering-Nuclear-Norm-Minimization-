import csv

import time
import numpy as np
from numpy import linalg as LA


user = 943 + 1
items = 1682 + 1


max_error = 0
min_error = 10


def nuclear_norm_solve(A, mask, mu): #y, m, lamda
    x = np.random.randint(5, size=(A.shape))
    for epoch in range(10):
        if epoch % 10 == 0:
            print("Epoch ", epoch)
        temp = A - np.multiply(mask, x)
        temp = x + temp
        f1,f2, f3 = np.linalg.svd(temp, full_matrices=False)

        y = soft(np.diag(f2), mu/2.0)


        x = np.dot(f1, np.dot(y, f3))
    return x.round()

def dataset(n):
    name1 = 'Dataset/u' + str(n) + '.base'
    name2 = 'Dataset/u' + str(n) + '.test'

    train = np.zeros((user,items))
    mask = np.zeros((user,items))
    test = []

    with open(name2, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            test.append([int(row[0]), int(row[1]), float(row[2])])

    with open(name1, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            u = int(row[0])
            i = int(row[1])
            rate = float(row[2])
            train[u][i] = rate
            mask[u][i] = 1

    return test, train, mask


def model(test, train, mask, mu):
    fill_matrix = nuclear_norm_solve(train, mask, mu)
    error = 0.0
    global max_error, min_error
    min_error = 100
    max_error = 0
    # print(fill_matrix)
    for data in test:
        temp = (data[2]-fill_matrix[data[0]][data[1]])
        temp = abs(temp)
        if temp < min_error:
            min_error = temp
        if temp > max_error:
            max_error = temp
        error += temp
    error /= len(test)
    return error


def k_fold():

    mus = [10, 0.1, 0.4, 0.6, 0.8, 1.0]
    for mu in mus:
        print("MU = ",mu)
        error = 0.0
        start = time.time()
        print("Iteration 1")
        test,train, mask = dataset(1)
        temp = model(test,train, mask, mu)

        temp /= (max_error - min_error)
        error += temp
        print(temp)
        end = time.time()
        print("Time taken = ", end - start)

        start = time.time()
        print("Iteration 2")
        test,train, mask = dataset(2)
        temp = model(test,train, mask, mu)
        temp /= (max_error - min_error)

        error += temp
        print(temp)
        end = time.time()
        print("Time taken = ", end - start)

        start = time.time()
        print("Iteration 3")
        test,train, mask = dataset(3)
        temp = model(test,train, mask, mu)
        temp /= (max_error - min_error)

        error += temp
        print(temp)
        end = time.time()
        print("Time taken = ", end - start)

        start = time.time()
        print("Iteration 4")
        test,train, mask = dataset(4)
        temp = model(test,train, mask, mu)
        error += temp
        temp /= (max_error - min_error)

        print(temp)
        end = time.time()
        print("Time taken = ", end - start)

        start = time.time()
        print("Iteration 5")
        test,train, mask = dataset(5)
        temp = model(test,train, mask, mu)
        temp /= (max_error - min_error)

        error += temp
        print(temp)
        end = time.time()
        print("Time taken = ", end - start)

        print("For mu = ",mu,"Error = ", error / 5)
    return error / 5

def soft(t, s):
    t = np.absolute(t)
    copy = t.copy()
    t -= s
    dim = (t.shape)
    for i in range(dim[0]):
        for j in range(dim[1]):
            if t[i][j] < 0:
                copy[i][j] = -1
            elif t[i][j] > 0:
                copy[i][j] = 1
            else:
                copy[i][j] = 0
            t[i][j] = abs(t[i][j])

    for i in range(dim[0]):
        for j in range(dim[1]):
            t[i][j] -= s
            if t[i][j] < 0:
                t[i][j] = 0
    ret = np.multiply(copy, t)
    return ret


k_fold()
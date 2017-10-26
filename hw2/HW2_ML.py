import pandas as pd
import numpy as np
import math
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from numpy.linalg import inv

X_train = pd.read_csv('X_train.csv', header=None)
X_train = np.asarray(X_train)
y_train0 = pd.read_csv('y_train.csv', header=None)
y_train0 = np.asarray(y_train0)
y_train = []
for item in y_train0:
    y_train.extend(item)
X_test = pd.read_csv('X_test.csv', header=None)
X_test = np.asarray(X_test)
y_test0 = pd.read_csv('y_test.csv', header=None)
y_test0 = np.asarray(y_test0)
y_test = []
for item in y_test0:
    y_test.extend(item)

# (a)
sum = 0
for y in y_train:
    sum += y
pi = float(sum)/float(4508)

b0_arr = []
b1_arr = []
for j in range(54):
    b0_up = 0
    b1_up = 0
    for i in range(4508):
        b0_up += float(X_train[i][j] * (1 - y_train[i]))
        b1_up += float(X_train[i][j] * y_train[i])
    b0 = b0_up / float(2732)
    b1 = b1_up / float(1776)
    b0_arr.append(b0)
    b1_arr.append(b1)

p0_arr = []
p1_arr = []
for j in range(54, 57):
    p0_down = 0
    p1_down = 0
    for i in range(4508):
        p0_down += float(math.log(X_train[i][j]) * (1 - y_train[i]))
        p1_down += float(math.log(X_train[i][j]) * y_train[i])
    p0 = float(2732) / p0_down
    p1 = float(1776) / p1_down
    p0_arr.append(p0)
    p1_arr.append(p1)

def classifier(test_x):
    num_row, num_col = test_x.shape
    result = []
    for i in range(num_row):
        y0_predict = float(math.pow(pi, 0) * math.pow(1 - pi, 1))
        y1_predict = float(pi)
        for j in range(num_col-3):
            y0_predict *= (float(math.pow(b0_arr[j], test_x[i][j]) * float(math.pow(1 - b0_arr[j], 1 - test_x[i][j]))))
            y1_predict *= (float(math.pow(b1_arr[j], test_x[i][j]) * float(math.pow(1 - b1_arr[j], 1 - test_x[i][j]))))
        for j in range(num_col-3, num_col):
            y0_predict *= (p0_arr[j-54] * float(math.pow(test_x[i][j], -(p0_arr[j-54] + 1))))
            y1_predict *= (p1_arr[j-54] * float(math.pow(test_x[i][j], -(p1_arr[j-54] + 1))))
        #print (y0_predict, y1_predict)
        if y0_predict > y1_predict:
            result.append(0)
        else:
            result.append(1)
    return result

result_predict = classifier(X_test)

a00 = 0
a01 = 0
a10 = 0
a11 = 0
for i in range(93):
    if y_test[i] == 0 and result_predict[i] == 0:
        a00 += 1
    if y_test[i] == 1 and result_predict[i] == 0:
        a10 += 1
    if y_test[i] == 0 and result_predict[i] == 1:
        a01 += 1
    if y_test[i] == 1 and result_predict[i] == 1:
        a11 += 1

a0 = [a00, a10]
a1 = [a01, a11]
t = Table([a0, a1])
print t
accuracy1 = float(a00+a11) / float(len(result_predict))
print accuracy1

# (b)stem plot

plt.figure(figsize=(16, 8))
plt.stem(b0_arr, linefmt='b-', markerfmt='bo', basefmt='r-', label='y=0')
plt.stem(b1_arr, linefmt='b-', markerfmt='ro', basefmt='r-', label='y=1')
plt.legend()
plt.show()


# (c)K-NN

def k_nn(k, test_x, test_y, train_x, train_y):
    result = []
    numrow_train, numcol_train = train_x.shape
    numrow_test, numcol_test = test_x.shape

    for i in range(numrow_test):
        distance = []
        for j in range(numrow_train):
            diff = np.absolute(test_x[i] - train_x[j])
            dist = np.sum(diff)
            distance.append(dist)
        distance = np.asarray(distance)
        idx = np.argpartition(distance, k)[:k]
        num0, num1 = 0, 0
        for h in idx:
            if train_y[h] == 0:
                num0 += 1
            if train_y[h] == 1:
                num1 += 1
        if num0 > num1:
            result.append(0)
        else:
            result.append(1)
    correct = 0
    for i in range(len(result)):
        if result[i] == test_y[i]:
            correct += 1
    accuracy = float(correct) / float(len(result))
    return accuracy

accuracy_arr = []
for k in range(1, 21):
    accuracy_arr.append(k_nn(k, X_test, y_test, X_train, y_train))

plt.plot(range(1, 21), accuracy_arr)
plt.xlabel('k')
plt.ylabel('Prediction accuracy')
plt.show()

# Change the dataset
X_train = np.pad(X_train, ((0, 0), (0, 1)), mode='constant', constant_values=1)
X_test = np.pad(X_test, ((0, 0), (0, 1)), mode='constant', constant_values=1)
for i in range(len(y_train)):
    if y_train[i] == 0:
        y_train[i] = -1
for i in range(len(y_test)):
    if y_test[i] == 0:
        y_test[i] = -1
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


# (d)Steepest Ascent Algorithm

def prob_right(x, y, w):
    x = x[np.newaxis].T
    x_T = np.transpose(x)
    a = y * x_T
    b = np.dot(a, w)
    return sigmoid(b)[0][0]

def steepest_ascent(x, y):
    w = np.zeros((58, 1))
    L = []
    for t in range(1, 10001):
        print t
        n = float(1) / float(100000 * math.pow(t + 1, 0.5))
        sum = np.zeros((58, 1))
        L1 = 0
        for i in range(4508):
            s1 = float(1 - prob_right(x[i], y[i], w)) * y[i]
            s2 = s1 * x[i][np.newaxis].T
            sum += s2
            if prob_right(x[i], y[i], w) == 0:
                L1 += print_x(x[i], y[i], w) - np.log(1+math.pow(math.e, print_x(x[i], y[i], w)))
            else:
                L1 += np.log(prob_right(x[i], y[i], w))
        L.append(L1)
        w += n * sum
    return L

plt.plot(range(1, 10001), steepest_ascent(X_train, y_train))
plt.xlim((-500, 10501))
plt.xlabel("iteration")
plt.ylabel("Objective training function")
plt.show()



def print_x(x,y,w):
    x = x[np.newaxis].T
    x_T = np.transpose(x)
    a = y * x_T
    b = np.dot(a, w)
    return b[0][0]

# (e) Newton method algorithm

def theta(x, w):
    x = x[np.newaxis].T
    x_T = np.transpose(x)
    b = np.dot(x_T, w)
    return sigmoid(b)[0][0]

def Newton_method(x, y):
    w = np.zeros((58, 1))
    L = []
    for t in range(1, 101):
        n = float(1) / float(math.pow(t+1, 0.5))
        triangle_w = np.zeros((58, 58))
        L1 = 0
        sum = np.zeros((58, 1))
        for i in range(4508):
            thet = theta(x[i], w)
            xi = x[i][np.newaxis].T
            xi_T = np.transpose(xi)
            a = float(thet * (1-thet)) * xi
            triangle_w -= np.dot(a, xi_T)

            s1 = float(1 - prob_right(x[i], y[i], w)) * y[i]
            s2 = s1 * x[i][np.newaxis].T
            sum += s2
            if prob_right(x[i], y[i], w) == 0:
                L1 += print_x(x[i], y[i], w) - np.log(1 + math.pow(math.e, print_x(x[i], y[i], w)))
            else:
                L1 += np.log(prob_right(x[i], y[i], w))
        L.append(L1)
        triangle_w_inv = inv(triangle_w)
        k = n * np.dot(triangle_w_inv, sum)
        w -= k
    return L

plt.plot(range(1, 101), Newton_method(X_train, y_train))
plt.show()


# Calculate accuracy

def get_w(x, y):
    w = np.zeros((58, 1))
    L = []
    for t in range(1, 101):
        n = float(1) / float(math.pow(t+1, 0.5))
        sum = np.zeros((58, 1))
        triangle_w = np.zeros((58, 58))
        L1 = 0
        sum = np.zeros((58, 1))
        for i in range(4508):
            thet = theta(x[i], w)
            xi = x[i][np.newaxis].T
            xi_T = np.transpose(xi)
            a = float(thet * (1-thet)) * xi
            triangle_w -= np.dot(a, xi_T)

            s1 = float(1 - prob_right(x[i], y[i], w)) * y[i]
            s2 = s1 * x[i][np.newaxis].T
            sum += s2
            if prob_right(x[i], y[i], w) == 0:
                L1 += print_x(x[i], y[i], w) - np.log(1 + math.pow(math.e, print_x(x[i], y[i], w)))
            else:
                L1 += np.log(prob_right(x[i], y[i], w))
        L.append(L1)
        triangle_w_inv = inv(triangle_w)
        k = n * np.dot(triangle_w_inv, sum)
        w -= k
    return w

parameter = get_w(X_train, y_train)

y_predict = []
for i in range(93):
    print i
    yi = np.dot(X_test[i], parameter)
    sigmoid1 = sigmoid(np.array([yi]))
    if sigmoid1[0][0] > 0.5:
        y_predict.append(1)
    else:
        y_predict.append(-1)

correct_num = 0
for i in range(len(y_predict)):
    if y_predict[i] == y_test[i]:
        correct_num += 1
accuracy = float(correct_num) / float(len(y_predict))
print accuracy

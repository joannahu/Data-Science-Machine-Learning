import csv
import numpy as np
import math
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from scipy.special import expit


#Import data
x_train=[]
y_train=[]
x_test=[]
y_test=[]

csvfile = file('X_train.csv', 'rb')
reader = csv.reader(csvfile)
for row in reader:
    x_train.append(row)

csvfile = file('Y_train.csv', 'rb')
reader = csv.reader(csvfile)
for row in reader:
    y_train.append(row)

csvfile = file('X_test.csv', 'rb')
reader = csv.reader(csvfile)
for row in reader:
    x_test.append(row)

csvfile = file('Y_test.csv', 'rb')
reader = csv.reader(csvfile)
for row in reader:
    y_test.append(row)
    
#convert to matrix
x_train=np.asarray(x_train,np.float)
y_train=np.asarray(y_train,np.float)
x_test=np.asarray(x_test,np.float)
y_test=np.asarray(y_test,np.float)

#get number of y_train
n_ytrain = len(y_train)

#get number of y_test
n_ytest = len(y_test)

#get number of features
n_row, n_feature = x_train.shape

#get number of y=0 and y=1
n_zero = 0
n_one = 0
for i in range(n_ytrain):
    if y_train[i][0]==0:
        n_zero += 1
    else:
        n_one += 1

#a
def get_pi():
    n_row, n_col = y_train.shape
    pi = 0
    for y in y_train:
        pi += y
    pi = float(pi)/n_ytrain
    return pi
    
def get_theta1():
    theta1_0=[0]*54 #array with size=54
    theta1_1=[0]*54
    for j in range(54):
        theta1_0[j] = 0
        theta1_1[j] = 0
        for i in range(n_ytrain):
            if y_train[i][0]==0:
                theta1_0[j] += float(x_train[i][j] * (1 - y_train[i]))
            else:
                theta1_1[j] += float(x_train[i][j] * y_train[i])
        theta1_0[j] = theta1_0[j]/n_zero
        theta1_1[j] = theta1_1[j]/n_one
    return theta1_0,theta1_1

def get_theta2():
    theta2_0=[0]*3 #array with size=3
    theta2_1=[0]*3
    for j in range(54,57):
        index=j-54
        theta2_0[index] = 0
        theta2_1[index] = 0
        for i in range(n_ytrain):
            if y_train[i][0]==0:
                theta2_0[index] += float(math.log(x_train[i][j]) * (1 - y_train[i]))
            else:
                theta2_1[index] += float(math.log(x_train[i][j]) * y_train[i])
        theta2_0[index] = n_zero/theta2_0[index]
        theta2_1[index] = n_one/theta2_1[index]
    return theta2_0,theta2_1

def bayes_classifier():
    pi = get_pi()
    theta1_y0,theta1_y1 = get_theta1()
    theta2_y0,theta2_y1 = get_theta2()
    result = []
    for i in range(n_ytest):
        y0_predict = float(1 - pi)
        y1_predict = float(pi)
        for j in range(54):
            y0_predict *= (float(math.pow(theta1_y0[j], x_test[i][j]) * float(math.pow(1 - theta1_y0[j], 1 - x_test[i][j]))))
            y1_predict *= (float(math.pow(theta1_y1[j], x_test[i][j]) * float(math.pow(1 - theta1_y1[j], 1 - x_test[i][j]))))
        for j in range(54, 57):
            y0_predict *= (theta2_y0[j-54] * float(math.pow(x_test[i][j], -(theta2_y0[j-54] + 1))))
            y1_predict *= (theta1_y1[j-54] * float(math.pow(x_test[i][j], -(theta1_y1[j-54] + 1))))
        if y0_predict > y1_predict:
            result.append(0)
        else:
            result.append(1)
    return result

def plot_bayes_accuracy():
    result = bayes_classifier()

    t00 = 0
    t01 = 0
    t10 = 0
    t11 = 0
    for i in range(n_ytest):
        if result[i] == 0 and y_test[i] == 0:
            t00 += 1
        elif result[i] == 0 and y_test[i] == 1:
            t01 += 1
        elif result[i] == 1 and y_test[i] == 0:
            t10 += 1
        else:
            t11 += 1
    table = PrettyTable(["/","actual y=0", "actual y=1"])
    table.align["/"] = "1"
    table.padding_width = 1  
    table.add_row(["predicted y=0", t00, t01])
    table.add_row(["predicted y=1", t10, t11])
    print(table)     
    
    accuracy = float(t00+t11) / len(result)
    print accuracy

plot_bayes_accuracy()

# b plot
theta1_y0,theta1_y1 = get_theta1()
plt.stem(theta1_y0, linefmt='y-', markerfmt='r.', basefmt='r-', label='y=0')
plt.stem(theta1_y1, linefmt='y-', markerfmt='b.', basefmt='b-', label='y=1')
plt.legend()
plt.show()

#c
def knn_classifier(k):
    result = []

    for i in range(n_ytest): #iterate all test data
        dist = [] #distance
        idx = [] #index of k-nearest neighbor
        for j in range(n_ytrain): # for each test data, iterate all training data
            dist.append(np.sum(np.absolute(x_test[i] - x_train[j])))
        
        idx = np.argpartition(dist, k) #get the index where the k smallest item is befroe the position k
        idx= idx[:k]#get smallest k items
        n_0 = 0
        n_1 = 0
        for j in idx:
            if y_train[j] == 0:
                n_0 += 1
            else:
                n_1 += 1
        if n_0 > n_1:
            result.append(0)
        else:
            result.append(1)
        #print dist[1]
    return result

def plot_knn_accuracy():
    #calculate accuracy of each k
    accuracy= []
    for k in range(1, 21):
        sum = 0
        result = knn_classifier(k)
        for i in range(len(result)):
            if result[i] == y_test[i]:
                sum += 1
        accuracy.append(float(sum) / len(result))
     
    #plot
    plt.plot(range(1, 21), accuracy)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.show()
    
plot_knn_accuracy()   

# d
#Transform data for logistic regression
x_train = np.pad(x_train, ((0, 0), (0, 1)), mode='constant', constant_values=1)
x_test = np.pad(x_test, ((0, 0), (0, 1)), mode='constant', constant_values=1)
for i in range(n_ytrain):
    if y_train[i] == 0:
        y_train[i] = -1
for i in range(n_ytest):
    if y_test[i] == 0:
        y_test[i] = -1

iterate_times = 10000
def steepest_ascent():
    w = np.zeros(n_feature+1)
    L = []

    for step in range(iterate_times):
        l = 0
        delta_L = 0
        for i in range(n_ytrain):
            x = y_train[i]*np.dot(x_train[i].T, w)
            sigmoid = expit(x)
            
            #calculate L
            if sigmoid == 0: #avoid log(0)
                l += (x - np.log(1+math.pow(math.e, x)))
            else:
                l += np.log(sigmoid)

            #update w
            delta_L += float(1 - sigmoid) * y_train[i] * x_train[i]
        
        L.append(l)
        step_size = float(1) / float(math.pow(10, 5) * math.pow(step + 1, 0.5))
        w += step_size*delta_L
    return L
           
def plot_logistic_regression():  
    plt.plot(range(0, iterate_times), steepest_ascent())
    plt.xlabel("iteration times ")
    plt.ylabel("value of objective function")
    plt.show()

plot_logistic_regression()
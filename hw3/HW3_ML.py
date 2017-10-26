import pandas as pd
import numpy as np
import math
from prettytable import PrettyTable
import matplotlib.pyplot as plt

'''Problem 1: PREDICTIONS WITH GAUSSIAN PROCESSES'''
#load data
X_train = pd.read_csv('gaussian_process/X_train.csv', header=None)
y_train = pd.read_csv('gaussian_process/y_train.csv', header=None)
X_test = pd.read_csv('gaussian_process/X_test.csv', header=None)
y_test = pd.read_csv('gaussian_process/y_test.csv', header=None)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

def cal_kernel_element(b, xi, xj):
    x = xi - xj
    return math.exp((-1/float(b)) * np.dot(x,x.transpose()))
            
train_size = len(X_train)
test_size = len(X_test)
K = np.zeros(shape=(train_size, train_size))
KD = np.zeros(shape=(test_size, train_size))
y_predict = np.zeros(shape=(test_size, 1))
RMSE = []

for b in (5,7,9,11,13,15): #loop b 
    for i in range (train_size):
        for j in range (train_size):#calculate K
            K[i][j] = cal_kernel_element(b, X_train[i], X_train[j])
        for j in range (test_size):#calculate K(x,D)
            KD[j][i] = cal_kernel_element(b, X_test[j], X_train[i])
    for sigma in (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1): #loop sigma
        for j in range (test_size): #calculate y_predict
            y_predict[j] = np.dot(np.dot(KD[j],np.linalg.inv(sigma*np.identity(train_size)+K)),y_train)
        RMSE.append(np.sqrt(np.sum(np.power((y_test-y_predict),2))/test_size))

table = PrettyTable(["RMSE"])
for i in range(len(RMSE)):
    table.add_row([RMSE[i]])
print table

x4_train = X_train[:,3]
x_y_plot = np.zeros(shape=(train_size, 2))
b = 5
sigma = 2
for i in range (train_size):
    for j in range (train_size):#calculate K
        K[i][j] = cal_kernel_element(b, x4_train[i], x4_train[j])
for j in range (train_size): #calculate y_predict
    x_y_plot[j] = [x4_train[j],np.dot(np.dot(K[j],np.linalg.inv(sigma*np.identity(train_size)+K)),y_train)]

fig1=plt.figure()
plt.scatter(x4_train.tolist(),np.hstack(y_train).tolist(),color="purple")
plt.xlabel("x_train[4]")
plt.ylabel('y_train')
plt.title("x_train[4] VS y_train ")
plt.grid()
x_y_plot = x_y_plot[x_y_plot[:,0].argsort()]
plt.plot(x_y_plot[:,0].tolist(),x_y_plot[:,1].tolist(),color="red")
#plt.axhline(y=np.mean(y_predict))
plt.show()

'''Problem 2: Least Square + Boosting'''
#load data
X_train = pd.read_csv('boosting/X_train.csv', header=None)
y_train = pd.read_csv('boosting/y_train.csv', header=None)
X_test = pd.read_csv('boosting/X_test.csv', header=None)
y_test = pd.read_csv('boosting/y_test.csv', header=None)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
train_size = len(X_train)
test_size = len(X_test)
X_train = np.c_[ np.ones(train_size), X_train] 
X_test = np.c_[ np.ones(test_size), X_test] 

def least_square_classifier(X_train, y_train):
    w = np.ones(shape=(len(X_train[0]), 1))
    w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train)),X_train.transpose()),y_train)
    return w

#initialize w (weight for choosing (xi,yi))
w = np.ones(shape=(train_size, 1))
w /= train_size

T = 1500
alpha = np.zeros(shape=(T, 1))
epsilon = np.zeros(shape=(T, 1))
X_train_B = np.zeros(shape=(train_size, len(X_train[0])))
y_train_B = np.zeros(shape=(train_size, 1))
param = np.zeros(shape=(T,len(X_train[0])))
f = np.zeros(shape=(train_size,1))
f_train = np.zeros(shape=(train_size,1))
f_test = np.zeros(shape=(test_size,1))
y_train_predict = np.zeros(shape=(train_size, 1))
y_test_predict = np.zeros(shape=(test_size, 1))
error_train = np.zeros(shape=(T, 1))
error_test = np.zeros(shape=(T, 1))
upper_bound = np.zeros(shape=(T, 1))
#select_times = np.zeros(shape=(train_size,1))
select_times = []
for t in range(T): #boosting for T = 1500 rounds
    #pick sample B
    B = np.random.choice(train_size, train_size, replace=True, p=np.hstack(w).tolist()) #choose train_size element from [0,train_size), choose with return
    p = 0
    select_times.extend(B.tolist())
    for i in B.tolist():
        X_train_B[p] = X_train[i]
        y_train_B[p] = y_train[i]
        p += 1
        #select_times[i] += 1
    
    #learn classifier
    param[t] = least_square_classifier(X_train_B,y_train_B).transpose()
    
    #update epsilon
    for i in range(train_size):
        f[i] = np.sign(np.dot(X_train[i], param[t].transpose())) 
        if f[i]!=y_train[i]: 
            epsilon[t] += w[i]
    if epsilon[t] > 0.5:
        param[t] *= (-1)
        #recalculate epsion
        epsilon[t] = 0
        for i in range(train_size):
            f[i] = np.sign(np.dot(X_train[i], param[t].transpose())) 
            if f[i]!=y_train[i]: 
                epsilon[t] += w[i]
        
    #update alpha
    alpha[t] = 0.5*np.log(float(1-epsilon[t])/float(epsilon[t])) 
    
    #update w
    for i in range(train_size): 
        w[i] *= math.exp(-alpha[t]*y_train[i]*f[i])
    sum = np.sum(w)
    for i in range(train_size):
        w[i] = float(w[i])/float(sum)
    
    #train error
    for i in range(train_size):
        f_train[i] += (alpha[t]*np.sign(np.dot(X_train[i], param[t])))
        y_train_predict[i] = np.sign(f_train[i])
        if y_train_predict[i]!=y_train[i]:
            error_train[t] += 1
    error_train[t] /= train_size

    #test error
    for i in range(test_size):
        f_test[i] += (alpha[t]*np.sign(np.dot(X_test[i], param[t])))
        y_test_predict[i] = np.sign(f_test[i])
        if y_test_predict[i]!=y_test[i]:
            error_test[t] += 1
    error_test[t] /= test_size
    
    #upper bound of training error
    if t>0:
        upper_bound[t] = upper_bound[t-1]*math.exp(-2*np.power((0.5-epsilon[t]),2))
    else:
        upper_bound[t] = math.exp(-2*np.power((0.5-epsilon[t]),2))

plt.plot(range(T),error_train,color="purple", lw=1)
plt.plot(range(T),error_test,color="red",lw=2)
plt.legend(["train","test"])
plt.xlabel("Step")
plt.ylabel('Error')
plt.title("Boosting steps VS Errors ")
plt.grid()
plt.show()

plt.plot(range(T),upper_bound,color="purple", lw=1)
plt.xlabel("Step")
plt.ylabel('Upper bound of training error')
plt.title("Boosting steps VS Upper bound of errors ")
plt.grid()
plt.show()

plt.hist(select_times,bins=1036)
plt.title("Select Times of Each Training Data")
plt.xlabel("Training data")
plt.ylabel("Times")
plt.show()

plt.plot(range(T),alpha,color="purple", lw=1)
plt.xlabel("Step")
plt.ylabel('alpha')
plt.title("Boosting steps VS Alpha ")
plt.grid()
plt.ylim((0,0.5))
plt.show()

plt.plot(range(T),epsilon,color="purple", lw=1)
plt.xlabel("Step")
plt.ylabel('Epsilon')
plt.title("Boosting steps VS Epsilon ")
plt.grid()
plt.ylim((0.3,0.6))
plt.show()

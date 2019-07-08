import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.io import loadmat
import utils
from scipy import optimize

###===================Part 1: Load and  Visualizing Data ============================
data = loadmat("D:/TJH/ML03/machine-learning-ex3/machine-learning-ex3/ex3/ex3data1.mat")
X = data["X"]
y = data["y"].ravel()
y[y==10] = 0
m = len(y)
sel = X[np.random.choice(m,100)]
utils.displayData(sel)
plt.show()

##====================Part 2a: Vectorize Logistic Regression ========================
def h(theta, x):
    z = np.matmul(x, theta)
    y = 1. / (1. + np.exp(-z))
    return y


def costFunction(theta, X, y, lamda):
    H = h(theta, X).T
    theta1 = theta.copy()
    theta1[0] = 0

    J = (-np.matmul(np.log(H), y) - np.matmul(np.log(1 - H), 1 - y)) / len(y) + lamda * (np.matmul(theta1,theta1))/ 2 / len(y)
    diff = (h(theta, X) - y)
    G = np.matmul(X.T, diff) / len(y) + lamda * theta1 / len(y)

    return J, G

theta_t = np.array([-2,-1,1,2])
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
y_t = np.array([1,0,1,0,1])
lambda_t = 3
J, grad = costFunction(theta_t,X_t,y_t,lambda_t)
print("Cost:{}".format(J))
print("Expected cost: 2.534819")
print("grad: {}".format(grad))
print("Excpected grad:  0.146561 -0.548558 0.724722 1.398003")

##====================Part 2b: One-vs-All Training ========================



def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape
    all_theta = np.zeros((num_labels,n+1))
    X = np.concatenate([np.ones((m,1)),X],axis=1)
    y_list = np.unique(y)

    options = {"maxiter": 400}
    initial_theta = np.zeros(n+1)

    for k in range(num_labels):
        K = y_list[k]
        res_k = optimize.minimize(costFunction,initial_theta,(X,y==K,lambda_),jac=True,method="TNC",options=options)
        theta_k = res_k.x
        all_theta[k] = theta_k

    return all_theta

lambda_ = 0.1
all_theta = oneVsAll(X,y,10,lambda_)
y_list = np.unique(y)

##======================Part3 prediction =================================

def precitOneVsAll(all_theta, X):
    m, n = X.shape
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    z = np.matmul(X,all_theta.T)
    h = 1./(1. + np.exp(-z))
    idx = np.argmax(h,axis=1)
    prediction = np.zeros(m)
    for j in range(m):
        prediction[j] = y_list[idx[j]]
    return prediction

prediction = precitOneVsAll(all_theta,X)
print("Training Set Accuracy:{:.2f}%".format((prediction==y).mean()*100))


##====================Part 4. Neural Network =====================

weights = loadmat("D:/TJH/ML03/machine-learning-ex3/machine-learning-ex3/ex3/ex3weights.mat")
theta1 = weights["Theta1"]
theta2 = weights["Theta2"]
theta2 = np.roll(theta2,1,axis=0)
print(theta1.shape,theta2.shape)

def sigmoid(z):
    return 1./(1.+ np.exp(-z))

def predict(theta1,theta2,X):
    m, n = X.shape
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    z1 = np.matmul(X,theta1.T)
    a1 = sigmoid(z1)
    a1 = np.concatenate([np.ones((m, 1)), a1], axis=1)
    z2 = np.matmul(a1,theta2.T)
    a2 = sigmoid(z2)

    prediction = np.argmax(a2,axis=1)

    return prediction


NN_prediction = predict(theta1,theta2,X)
print(("Training Set Accuracy:{:.2f}%".format((NN_prediction==y).mean()*100)))

sample = X[-102:-101,:]
utils.displayData(sample, figsize=(4, 4))
pre = predict(theta1,theta2,sample)
print(pre[0])
plt.show()











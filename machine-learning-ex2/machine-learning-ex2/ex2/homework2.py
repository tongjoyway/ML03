import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


###====================part1: plotting ==================
data = pd.read_csv("ex2data1.txt",sep=",",header=None)
pos = data[data[2]==1]
neg = data[data[2]==0]
plt.scatter(pos[0],pos[1],marker="+")
plt.scatter(neg[0],neg[1],marker="o")
plt.show()


###========================Part2: compute cost and Gradient===============
X = data.iloc[:,[0,1]]
y = data.iloc[:,-1]
m,n = X.shape
ones = pd.Series([1.]*m)
X = pd.concat([ones,X],axis=1)
X = np.array(X)
y = np.array(y).reshape(-1,1)

initial_theta = np.zeros((n+1,1))

def h(theta,x):
    z = np.matmul(x,theta)
    y =  1./(1. + np.exp(-z))
    return y

def costFunction(theta,X,y):
    # for i in range(m):
    #     h_i = h(theta,X[i:i+1])[0][0]
    #     y_i = y[i][0]
    #     J +=  (-y_i*np.log(h_i)-(1-y_i)*np.log(1-h_i))
    # J /= m
    H = h(theta,X).T
    J = (-np.matmul(np.log(H),y)-np.matmul(np.log(1-H),1-y))[0][0]/len(y)

    diff = (h(theta,X) - y)
    G = np.matmul(X.T,diff)/len(y)

    return J, G

cost,grad = costFunction(initial_theta,X,y)
print("Cost at inital theta(zeros):%f"%cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):')
print("%s"%(grad.T[0]))
print('Expected gradients (approx): -0.1000 -12.0092 -11.2628\n')

test_theta = np.array([[-24],[0.2],[0.2]])
cost, grad = costFunction(test_theta,X,y)
print("Cost at inital theta(zeros):%f"%cost)
print('Expected cost (approx): 0.218')
print('Gradient at initial theta (zeros):')
print("%s"%(grad.T[0]))
print('Expected gradients (approx): 0.043 2.566 2.647\n')


def gradientDescent(X,y,initial_theta,alpha,iterations):
    theta = initial_theta
    J = 0
    for i in range(iterations):
        alphaa = alpha
        if i > iterations*0.5:
            alphaa = alpha * 0.2
        elif i > iterations * 0.7:
            alphaa = alpha * 0.04
        J,grad = costFunction(theta,X,y)
        theta -= alphaa * grad
    return theta, J

theta,cost = (gradientDescent(X,y,initial_theta,0.005,500000))
print("Cost at theta found by fminunc: %f"%(cost))
print("Expected cost(approx): 0.203")
print("theta:%s"%theta.T[0])
print("Expected theta(approx): -25.161, 0.206, 0.201")

###==================Part 3: Optimizing and plot ======================

def plotDecisionBoundary(theta,X,y):
    y = y.T[0]
    pos_index = y == 1
    neg_index = y == 0
    pos_data = X[pos_index]
    neg_data = X[neg_index]
    plt.scatter(pos_data[:,1], pos_data[:,2], marker="+",label="admitted")
    plt.scatter(neg_data[:,1], neg_data[:,2], marker="o",label="not admitted")
    xx = np.linspace(0,100,100)
    yy = -theta[0]/theta[2] - (theta[1]/theta[2])*xx
    plt.plot(xx,yy,label="decision boundary")
    plt.xlim((30,100))
    plt.ylim((30,100))
    plt.legend()
    plt.show()

plotDecisionBoundary(theta,X,y)

###===============Part 4: Predict and Accuracies=========================

prob = h(theta,np.array([[1,45,85]]))[0][0]
print("For a student with scores 45 and 85, we predict an admission probability of %f"%prob)
print("Expected value: 0.775 +/- 0.002")


p = h(theta,X).T[0]
p = np.round(p)
print("train accuracy: %f"%(np.mean(p==y.T)*100))
print("expected accuracy(approx):89.0")

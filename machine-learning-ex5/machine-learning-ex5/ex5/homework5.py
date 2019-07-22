import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.io import loadmat
import utils
from scipy import optimize

###===================Part 1.1: Load and  Visualizing Data ============================
data = loadmat("D:/TJH/ML03/machine-learning-ex5/machine-learning-ex5/ex5/ex5data1.mat")
X, y = data["X"], data["y"].ravel()   ### X: (12,1)
Xtest, ytest = data["Xtest"], data["ytest"].ravel()
Xval, yval = data["Xval"], data["yval"].ravel()

m = len(y)

# plt.plot(X, y, "ro", ms=10, mec="k", mew=1)
# plt.xlabel("Change in water level(x)")
# plt.ylabel("Water flowing out of the dam(y)")
# plt.show()

###===================Part 1.2 1.3 Regularized linear regression cost function and Gradient =============

def LinearRegCostFunction(X, y, theta, lambda_ =0.0):
    m = X.shape[0]
    h = np.dot(X,theta)
    theta_ = theta.copy()
    theta_[0] = 0
    J = (0.5/m) * ((h - y)**2).sum() + lambda_ * (0.5/m)*(theta_**2).sum()
    grad = (1./m)*np.dot(h-y,X) + lambda_ * (1/m) * theta_

    return J, grad

initial_theta = np.array([1,1])
X_ = np.concatenate([np.ones((m, 1)), X], axis=1)
J ,grad = LinearRegCostFunction(X_, y, initial_theta, lambda_=1.0)
print("Cost is: {:.6f}".format(J))
print("Gradient is: [{:.6f},{:.6f}]".format(*grad))   ### passed!

###===================Part 1.4 Fitting linear regression ===========================================
X_ = np.concatenate([np.ones((m, 1)), X], axis=1)
optimized_theta = utils.trainLinearReg(LinearRegCostFunction,X_,y,lambda_=0.)

# plt.plot(X, y, "ro", ms=10, mec="k", mew=1)
# plt.plot(X,np.dot(X_,optimized_theta),"--",lw=2)   ### passed!
# plt.show()

###=================== Part 2.1 Learning Curves ===================================================
def learningCurve(X, y, Xval, yval, lambda_=0):
    m = len(y)
    error_train = np.zeros(m)
    error_val   = np.zeros(m)
    Xval_aug = np.concatenate([np.ones((Xval.shape[0],1)),Xval],axis=1)

    for i in range(m):
        X_i = X[:i+1]
        X_i_aug = np.concatenate([np.ones((X_i.shape[0],1)),X_i],axis=1)
        y_i = y[:i+1]
        theta_i = utils.trainLinearReg(LinearRegCostFunction,X_i_aug,y_i,lambda_=lambda_)
        error_train[i] = LinearRegCostFunction(X_i_aug,y_i,theta_i,lambda_=0)[0]
        if i==0:
            error_val[i] = np.nan
        else:
            error_val[i]   = LinearRegCostFunction(Xval_aug,yval,theta_i,lambda_=0)[0]

    return error_train, error_val

error_train, error_val = learningCurve(X, y, Xval, yval, lambda_=0)
# plt.plot(np.linspace(0,m-1,m),error_train)
# plt.plot(np.linspace(0,m-1,m),error_val)
# plt.show()     #### passed!

###=================== Part 3 Polynomial Regression ===================================================
def polyFeatures(X,p):
    X_poly = np.zeros((X.shape[0],p))
    for i in range(X.shape[0]):
        for j in range(p):
            X_poly[i,j] = (X[i])**(j+1)
    return X_poly

p = 8
X_poly = polyFeatures(X,p)
X_poly, mu, sigma = utils.featureNormalize(X_poly)
X_poly = np.concatenate([np.ones((X_poly.shape[0],1)),X_poly],axis=1)

X_poly_test = polyFeatures(Xtest,p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.concatenate([np.ones((X_poly_test.shape[0],1)),X_poly_test],axis=1)

X_poly_val = polyFeatures(Xval,p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.concatenate([np.ones((X_poly_val.shape[0],1)),X_poly_val],axis=1)

print(X_poly[0,:])

###=================== Part 3.1 Learning Polynomial Regression ==========================
lambda_= 0.00
theta_lamda0 = utils.trainLinearReg(LinearRegCostFunction,X_poly,y,lambda_=lambda_)
plt.plot(X, y, "ro", ms=10, mec="k", mew=1)
utils.plotFit(polyFeatures,np.min(X),np.max(X),mu,sigma,theta_lamda0,p)
plt.ylim([-20, 50])
plt.show()

error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
plt.plot(np.linspace(0,m-1,m), error_train, np.linspace(0,m-1,m), error_val)
plt.show()



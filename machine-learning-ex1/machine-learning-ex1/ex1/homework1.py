import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


data = pd.read_csv("ex1data1.txt",sep=",",header=None)
X = data.iloc[:,0]
y = data.iloc[:,1]
m = len(y)

# plt.plot(X,y,"rx")
# plt.ylabel("Profit in $10,000")
# plt.xlabel("Polulation of City in 10,000s")
# plt.show()

ones = pd.Series([1.]*m)
X = pd.concat([ones,X],axis=1)
X = np.array(X)
y = np.array(y).reshape(-1,1)
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01

def computeCost(x,y,theta):
    m = len(y)
    J = 0
    diff = y - np.matmul(x,theta)
    for i in range(m):
        J += diff[i][0]**2
    return J/2./m

print(computeCost(X,y,theta))
print("You should expect to see a cost of 32.07")
print(computeCost(X,y,np.array([[-1.],[2.]])))
print("You should expect to see a cost of 54.24")

def gradientDescent(X,y,theta,alpha,iterations):
    m = len(y)
    for i in range(iterations):
        step1 = ((np.matmul(X,theta) - y))
        step2 = np.matmul(X.T,step1)
        step3 = step2/m
        theta -= alpha * step3
    return theta


THETA = (gradientDescent(X,y,theta,alpha,iterations))
print("theta1 and theta2 are %f and %f"%(THETA[0][0],THETA[1][0]))

predict1 = np.matmul(np.array([[1,3.5]]),THETA)[0][0]*10000
predict2 = np.matmul(np.array([[1,7.]]),THETA)[0][0]*10000
print("The prediction1 and 2 are %f and %f"%(predict1,predict2))

theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,100)
J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))


for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[j,i] = computeCost(X,y,t)


fig = plt.figure()
ax = fig.gca(projection= '3d')
surf = ax.plot_surface(theta0_vals,theta1_vals,J_vals)
plt.show()
plt.contour(theta0_vals,theta1_vals,J_vals,np.logspace(-2,3,50))
plt.show()
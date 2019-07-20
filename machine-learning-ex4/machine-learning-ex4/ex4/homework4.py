import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.io import loadmat
import utils
from scipy import optimize

###===================Part 1.1: Load and  Visualizing Data ============================
data = loadmat("D:/TJH/ML03/machine-learning-ex4/machine-learning-ex4/ex4/ex4data1.mat")
X, y = data["X"], data["y"].ravel()
y[y==10] = 0
m, n = X.shape

rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]
# utils.displayData(sel)
# plt.show()

###===================Part 1.2: Model Representation ============================
weights = loadmat("D:/TJH/ML03/machine-learning-ex4/machine-learning-ex4/ex4/ex4weights.mat")
theta1,theta2 = weights["Theta1"], weights["Theta2"]
theta2 = np.roll(theta2,1,axis=0)

input_layer_size = theta1.shape[1] - 1
hidden_layer_size = theta1.shape[0]
output_layer_size = theta2.shape[0]

nn_params = np.concatenate((theta1.ravel(),theta2.ravel()))


###===================Regularized costFunction============================
def sigmoidGradient(z):
    return utils.sigmoid(z)*(1.-utils.sigmoid(z))


def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,output_layer_size,X,y,lambda_=0.0):
    t1 = nn_params[:(input_layer_size + 1) * hidden_layer_size].reshape(hidden_layer_size,input_layer_size+1)
    t2 = nn_params[(input_layer_size + 1) * hidden_layer_size:].reshape(output_layer_size,hidden_layer_size+1)
    m = X.shape[0]
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    z2 = (np.dot(X,t1.T))
    a2 = utils.sigmoid(z2)   ### 5000*25
    z3 = (np.dot(np.concatenate([np.ones((m,1)),a2], axis=1),t2.T))
    h = utils.sigmoid(z3)

    y_matrix = np.zeros((m,output_layer_size))
    for i in range(m):
        y_matrix[i,y[i]] = 1.

    ####regularized costFunction
    J = 1.*(1/m)*(-y_matrix*np.log(h) - (1-y_matrix)*np.log(1-h)).sum(axis=1).sum() \
        + lambda_*(0.5/m)*((t1**2)[:,1:].sum().sum()+(t2**2)[:,1:].sum().sum())

    ###back propagation
    delta3 = h - y_matrix   ##5000*10   t2: 10*26  t1:25*401
    delta2 = np.dot(delta3,t2)*np.concatenate([np.zeros((m,1)),a2*(1-a2)], axis=1)  ## 5000*26

    grad2 = (1/m) * np.dot(delta3.T,np.concatenate([np.ones((m,1)),a2], axis=1)) \
            + (lambda_/m)*np.concatenate([np.zeros((t2.shape[0],1)),t2[:,1:]],axis=1)
    grad1 = (1/m) * np.dot(delta2.T[1:],X) \
            + (lambda_/m)*np.concatenate([np.zeros((t1.shape[0],1)),t1[:,1:]],axis=1)
    grad = np.concatenate((grad1.ravel(),grad2.ravel()))

    return J, grad

J_ = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,output_layer_size,X,y,lambda_=3)
print(J_)

###===================Part 2.3: Random Initialization============================
def randInitializeWeights(L_in,L_out,epsilon_init=0.12):
    W = np.random.rand(L_out,1+L_in)*2.*epsilon_init - epsilon_init
    return W


initial_theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size,output_layer_size)
initial_nn_params = np.concatenate((initial_theta1.ravel(),initial_theta2.ravel()))
###=================== Part 2.4 Gradient checking ===============================
utils.checkNNGradients(nnCostFunction, lambda_=0)   #######Relative Difference: 2.27415e-11，passed!!!


###================== Part 2.6 Learning Parameters using scipy.optimize.minimize ============
options = {"maxiter":100}
lamda_ = 1
costFunction = lambda p: nnCostFunction(p,input_layer_size ,hidden_layer_size,output_layer_size,X,y,lambda_=lamda_)

res = optimize.minimize(costFunction,initial_nn_params,jac=True,method="TNC",options=options)
optimized_nn_params = res.x
optimized_Theta1 = optimized_nn_params[:(input_layer_size + 1) * hidden_layer_size].reshape(hidden_layer_size,input_layer_size+1)
optimized_Theta2 = optimized_nn_params[(input_layer_size + 1) * hidden_layer_size:].reshape(output_layer_size,hidden_layer_size+1)

pred = utils.predict(optimized_Theta1,optimized_Theta2,X)
print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))  #####Training Set Accuracy: 96.420000 passed!!!

utils.displayData(optimized_Theta1[:,1:])
plt.show()

# nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,output_layer_size,X,y)

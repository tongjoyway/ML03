import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from scipy.io import loadmat
import utils
from scipy import optimize

### ===================== 1. Anomaly Detection =============================
data = loadmat("D:/TJH/ML03/machine-learning-ex8/machine-learning-ex8/ex8/ex8data1.mat")
X, Xval, yval = data['X'], data['Xval'], data['yval'][:, 0]
#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bx', mew=2, mec='k', ms=6)
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

def estimateGaussian(X):
    m, n = X.shape
    # You should return these values correctly
    mu = np.zeros(n)
    sigma2 = np.zeros(n)
    # ====================== YOUR CODE HERE ======================
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma2 = sigma ** 2
    # =============================================================
    return mu, sigma2

mu, sigma2 = estimateGaussian(X)
p = utils.multivariateGaussian(X, mu, sigma2)
#  Visualize the fit
utils.visualizeFit(X,  mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.tight_layout()
plt.show()

### ================== 1.3 Selecting the threshold, e ================
def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    for epsilon in np.linspace(1.01 * min(pval), max(pval), 1000):
        # ====================== YOUR CODE HERE =======================
        predictions = (pval < epsilon).astype(int)
        tp = ((predictions == 1)&(yval == 1)).sum()
        fp = ((predictions == 1)&(yval == 0)).sum()
        fn = ((predictions == 0)&(yval == 1)).sum()
        prec = tp/(tp + fp)
        rec = tp/(tp + fn)
        if prec == 0 and rec == 0:
            F1 = 0
        else:
            F1 = 2 * prec * rec/(prec + rec)

        # =============================================================
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
        # print(epsilon,F1,prec,rec,bestF1)
    return bestEpsilon, bestF1

pval = utils.multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval,pval)
print('Best epsilon found using cross-validation: %.2e' % epsilon)
print('Best F1 on Cross Validation Set:  %f' % F1)
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)')
#  Find the outliers in the training set and plot the
outliers = p < epsilon

#  Visualize the fit
utils.visualizeFit(X,  mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.tight_layout()
#  Draw a red circle around those outliers
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', ms=10, mfc='None', mew=2)
plt.show()

### =================== 1.4 High dimensional dataset ================
data = loadmat("D:/TJH/ML03/machine-learning-ex8/machine-learning-ex8/ex8/ex8data2.mat")
X, Xval, yval = data['X'], data['Xval'], data['yval'][:, 0]

mu, sigma2 = estimateGaussian(X)
p = utils.multivariateGaussian(X, mu, sigma2)
pval = utils.multivariateGaussian(Xval, mu, sigma2)
print(pval.min(),pval.max())
epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: %.2e' % epsilon)
print('Best F1 on Cross Validation Set          : %f\n' % F1)
print('  (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of      0.615385)')
print('\n# Outliers found: %d' % np.sum(p < epsilon))  #### Passed!!!


### =================== 2 Recommender Systems ================
data = loadmat("D:/TJH/ML03/machine-learning-ex8/machine-learning-ex8/ex8/ex8_movies.mat")
Y, R = data['Y'], data['R']
print('Average rating for movie 1 (Toy Story): %f / 5' %
      np.mean(Y[0, R[0, :] == 1]))

# We can "visualize" the ratings matrix by plotting it with imshow
# plt.figure(figsize=(8, 8))
# plt.imshow(Y)
# plt.ylabel('Movies')
# plt.xlabel('Users')
# plt.grid(False)
# plt.show()

def cofiCostFunc(params, Y, R, num_users, num_movies,
                      num_features, lambda_=0.0):
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)

    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    # ====================== YOUR CODE HERE ======================
    predict_y = np.dot(X,Theta.T)
    error2 = (Y - predict_y)**2
    for i in range(num_movies):
        for j in range(num_users):
            if R[i,j]==1:
                J += error2[i,j]
    J *= 0.5
    J += lambda_*0.5*(params**2).sum()

    error = predict_y - Y
    X_grad = np.dot(error*R,Theta) + lambda_ * X
    Theta_grad = np.dot((error*R).T,X) + lambda_ * Theta

    # =============================================================

    grad = np.concatenate([X_grad.ravel(), Theta_grad.ravel()])
    return J, grad

data = loadmat("D:/TJH/ML03/machine-learning-ex8/machine-learning-ex8/ex8/ex8_movieParams.mat")
X, Theta, num_users, num_movies, num_features = data['X'],\
        data['Theta'], data['num_users'], data['num_movies'], data['num_features']

#  Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, 0:num_users]
R = R[:num_movies, 0:num_users]

#  Evaluate cost function
J, _ = cofiCostFunc(np.concatenate([X.ravel(), Theta.ravel()]),
                    Y, R, num_users, num_movies, num_features,1.5)

print('Cost at loaded parameters:  %.2f \n(this value should be about 31.34)' % J)
utils.checkCostFunction(cofiCostFunc,1.5)  ##### passed

#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
movieList = utils.loadMovieList()
n_m = len(movieList)

#  Initialize my ratings
my_ratings = np.zeros(n_m)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
# Note that the index here is ID-1, since we start index from 0.
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print('New user ratings:')
print('-----------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d stars: %s' % (my_ratings[i], movieList[i]))


#  Now, you will train the collaborative filtering model on a movie rating
#  dataset of 1682 movies and 943 users

#  Load data
data = loadmat("D:/TJH/ML03/machine-learning-ex8/machine-learning-ex8/ex8/ex8_movies.mat")
Y, R = data['Y'], data['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
#  943 users

#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.hstack([my_ratings[:, None], Y])
R = np.hstack([(my_ratings > 0)[:, None], R])

#  Normalize Ratings
Ynorm, Ymean = utils.normalizeRatings(Y, R)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.concatenate([X.ravel(), Theta.ravel()])

# Set options for scipy.optimize.minimize
options = {'maxiter': 100}

# Set Regularization
lambda_ = 10
res = optimize.minimize(lambda x: cofiCostFunc(x, Ynorm, R, num_users,
                                               num_movies, num_features, lambda_),
                        initial_parameters,
                        method='TNC',
                        jac=True,
                        options=options)
theta = res.x

# Unfold the returned theta back into U and W
X = theta[:num_movies*num_features].reshape(num_movies, num_features)
Theta = theta[num_movies*num_features:].reshape(num_users, num_features)

print('Recommender system learning completed.')


p = np.dot(X, Theta.T)
my_predictions = p[:, 0] + Ymean

movieList = utils.loadMovieList()

ix = np.argsort(my_predictions)[::-1]

print('Top recommendations for you:')
print('----------------------------')
for i in range(10):
    j = ix[i]
    print('Predicting rating %.1f for movie %s' % (my_predictions[j], movieList[j]))

print('\nOriginal ratings provided:')
print('--------------------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s' % (my_ratings[i], movieList[i]))
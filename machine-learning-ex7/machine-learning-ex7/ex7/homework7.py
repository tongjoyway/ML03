import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from scipy.io import loadmat
import utils
from scipy import optimize

### ===================== 1 K-means Clustering =====================


def findClosestCentroids(X, centroids):
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0], dtype=int)

    # ====================== YOUR CODE HERE ======================
    for i in range(X.shape[0]):
        data_i = X[i]
        label = 0
        distance = np.linalg.norm((data_i - centroids[0]))
        for j in range(K):
            distance_j = np.linalg.norm((data_i - centroids[j]))
            if distance_j < distance:
                label = j
                distance = distance_j
        idx[i] = label
    # =============================================================
    return idx

data = loadmat("D:/TJH/ML03/machine-learning-ex7/machine-learning-ex7/ex7/ex7data2.mat")
X = data["X"]
K = 3
initial_centroids = np.array([[3,3],[6,2],[8,5]])
idx = findClosestCentroids(X,initial_centroids)
print(idx[:3])   ####   PASSED!


def computeCentroids(X, idx, K):
    m, n = X.shape
    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))

    # ====================== YOUR CODE HERE ======================
    for k in range(K):
        data_k = X[idx == k]
        centroids[k] = data_k.mean(axis = 0)
    # =============================================================
    return centroids

centroids = computeCentroids(X,idx,K)
print(centroids)   ### PASSED!


data = loadmat("D:/TJH/ML03/machine-learning-ex7/machine-learning-ex7/ex7/ex7data2.mat")
K = 3
max_iters = 10
initial_centroids = np.array([[3,3],[6,2],[8,5]])

# centroids, idx, anim = utils.runkMeans(X,initial_centroids,findClosestCentroids,computeCentroids,max_iters,True)
# anim
# plt.show()     #### passed!

def kMeansInitCentroids(X, K):
    m, n = X.shape
    # You should return this values correctly
    centroids = np.zeros((K, n))
    # ====================== YOUR CODE HERE ======================
    randidx = np.random.permutation(X)
    centroids = randidx[:K]
    # =============================================================
    return centroids

# ###======================= 下面不需要code ===========================================
# # ======= Experiment with these parameters ================
# # You should try different values for those parameters
# K = 16
# max_iters = 10
#
# # Load an image of a bird
# # Change the file name and path to experiment with your own images
# A = mpl.image.imread("D:/TJH/ML03/machine-learning-ex7/machine-learning-ex7/ex7/bird_small.png")
# # ==========================================================
#
# # Divide by 255 so that all values are in the range 0 - 1
# A /= 255
#
# # Reshape the image into an Nx3 matrix where N = number of pixels.
# # Each row will contain the Red, Green and Blue pixel values
# # This gives us our dataset matrix X that we will use K-Means on.
# X = A.reshape(-1, 3)
#
# # When using K-Means, it is important to randomly initialize centroids
# # You should complete the code in kMeansInitCentroids above before proceeding
# initial_centroids = kMeansInitCentroids(X, K)
#
# # Run K-Means
# centroids, idx = utils.runkMeans(X, initial_centroids,
#                                  findClosestCentroids,
#                                  computeCentroids,
#                                  max_iters)
#
# # We can now recover the image from the indices (idx) by mapping each pixel
# # (specified by its index in idx) to the centroid value
# # Reshape the recovered image into proper dimensions
# X_recovered = centroids[idx, :].reshape(A.shape)
#
# # Display the original image, rescale back by 255
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].imshow(A*255)
# ax[0].set_title('Original')
# ax[0].grid(False)
#
# # Display compressed image, rescale back by 255
# ax[1].imshow(X_recovered*255)
# ax[1].set_title('Compressed, with %d colors' % K)
# ax[1].grid(False)
# plt.show()

### ===================== 2 Principal Component Analysis ===================
data = loadmat("D:/TJH/ML03/machine-learning-ex7/machine-learning-ex7/ex7/ex7data1.mat")
X = data["X"]
def pca(X):
    m, n = X.shape
    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)
    # ====================== YOUR CODE HERE ======================
    sigma = X.T.dot(X)/m
    U, S, V = np.linalg.svd(sigma)
    # ============================================================
    return U, S

X_norm, mu, sigma = utils.featureNormalize(X)
U, S = pca(X_norm)
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=0.25)

for i in range(2):
    ax.arrow(mu[0], mu[1], 1.5 * S[i]*U[0, i], 1.5 * S[i]*U[1, i],
             head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)

ax.axis([0.5, 6.5, 2, 8])
ax.set_aspect('equal')
ax.grid(False)
plt.show()
print('Top eigenvector: U[:, 0] = [{:.6f} {:.6f}]'.format(U[0, 0], U[1, 0]))
print(' (you should expect to see [-0.707107 -0.707107])')    ### PASSED!

###================2.3 Dimensionality Reduction with PCA ==================
def projectData(X, U, K):
    Z = np.zeros((X.shape[0], K))
    # ====================== YOUR CODE HERE ======================
    Z = np.dot(X,U[:,:K])
    # =============================================================
    return Z
K = 1
Z = projectData(X_norm, U, K)
print(Z[0,0])  #### 1.481274  Passed!

def recoverData(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    # ====================== YOUR CODE HERE ======================
    X_rec = np.dot(Z,U[:,:K].T)
    # =============================================================
    return X_rec

X_rec = recoverData(Z, U, K)
print(X_rec[0,0],X_rec[0,1]) ### [-1.04741883 -1.04741883] passed!
#  Plot the normalized dataset (returned from featureNormalize)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(X_norm[:, 0], X_norm[:, 1], 'bo', ms=8, mec='b', mew=0.5)
ax.set_aspect('equal')
ax.grid(False)
plt.axis([-3, 2.75, -3, 2.75])

# Draw lines connecting the projected points to the original points
ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', mec='r', mew=2, mfc='none')
for xnorm, xrec in zip(X_norm, X_rec):
    ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)
# plt.show()    PASSED!

### ===================== 2.4 Face Image Dataset =====================
data = loadmat("D:/TJH/ML03/machine-learning-ex7/machine-learning-ex7/ex7/ex7faces.mat")
X = data['X']
utils.displayData(X[:100, :], figsize=(8, 8))
plt.show()

#  normalize X by subtracting the mean value from each feature
X_norm, mu, sigma = utils.featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Visualize the top 36 eigenvectors found
utils.displayData(U[:, :36].T, figsize=(8, 8))
plt.show()

#  Project images to the eigen space using the top k eigenvectors
#  If you are applying a machine learning algorithm
K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a shape of: ', Z.shape)

#  Project images to the eigen space using the top K eigen vectors and
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed
K = 100
X_rec  = recoverData(Z, U, K)

# Display normalized data
utils.displayData(X_norm[:100, :], figsize=(6, 6))
plt.gcf().suptitle('Original faces')

# Display reconstructed data from only k eigenfaces
utils.displayData(X_rec[:100, :], figsize=(6, 6))
plt.gcf().suptitle('Recovered faces')
plt.show()   ############# Passed!
pass
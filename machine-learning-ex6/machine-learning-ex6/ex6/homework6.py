import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.io import loadmat
import utils
from scipy import optimize
import re

###===================Part 1.1: Load and  Visualizing Data ============================
data = loadmat("D:/TJH/ML03/machine-learning-ex6/machine-learning-ex6/ex6/ex6data1.mat")
X, y = data["X"], data["y"].ravel()   ### X: (12,1)

# utils.plotData(X,y)
# plt.show()

C = 1
# C = 100
model = utils.svmTrain(X,y,C,utils.linearKernel,1e-3,20)
# utils.visualizeBoundaryLinear(X,y,model)
# plt.show()

###===================Part 1.2.1: SVM with Gaussian Kernels ============================
def gaussianKernel(x1, x2, sigma):
    sim = 0
    # ====================== YOUR CODE HERE ======================
    norm = np.linalg.norm(x1 - x2)
    sim = np.exp(-norm**2/2/(sigma**2))
    # =============================================================
    return sim

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

sim = gaussianKernel(x1, x2, sigma)
print("this value is {:.6f}".format(sim))  ##pass

###===================Part 1.2.2: Example dataset2 ============================
data = loadmat("D:/TJH/ML03/machine-learning-ex6/machine-learning-ex6/ex6/ex6data2.mat")
X, y = data["X"], data["y"].ravel()   ### X: (12,1)

# utils.plotData(X,y)
# plt.show()

C = 1
sigma = 0.1
# model = utils.svmTrain(X,y,C,gaussianKernel,args=(sigma,))
# utils.visualizeBoundary(X,y,model)
# plt.show()   ### pass!

###===================Part 1.2.3: Example dataset3 ============================
data = loadmat("D:/TJH/ML03/machine-learning-ex6/machine-learning-ex6/ex6/ex6data3.mat")
X, y, Xval, yval = data["X"], data["y"].ravel(), data["Xval"], data["yval"].ravel()
# utils.plotData(X,y)
# plt.show()


def dataset3Params(X, y, Xval, yval):
    C = 0
    sigma = 0
    correctness = 0
    grid_list = [0.01,0.03,0.1,0.3,1,3,10,30]
    for C_i in grid_list:
        for sigma_i in grid_list:
            model_i = utils.svmTrain(X, y,C_i,gaussianKernel,args=(sigma_i,))
            predictions_i = utils.svmPredict(model_i,Xval)
            correctness_i = np.mean(predictions_i == yval)
            if correctness_i > correctness:
                C = C_i
                sigma = sigma_i
                correctness = correctness_i
            print(C,sigma,correctness_i)
    return C, sigma

# C, sigma = dataset3Params(X,y,Xval,yval)  ### C = 1, sigma = 0.1
# model = utils.svmTrain(X,y,C,gaussianKernel,args=(sigma,))
# utils.visualizeBoundary(X,y,model)
# plt.show()   ### pass!


###===================Part 2: Spam Classification ============================
###===================Part 2.1: Preprocessing Emails ============================
file = open("D:/TJH/ML03/machine-learning-ex6/machine-learning-ex6/ex6/emailSample1.txt")
email_contents = file.read()

def processEmail(email_contents,verbose=True):
    vocabList = utils.getVocabList()
    word_indices = []

    email_contents = email_contents.lower()
    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)
    email_contents = [word for word in email_contents if len(word) > 0]
    stemmer = utils.PorterStemmer()
    processed_email = []
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)

        if len(word) < 1:
            continue

        # Look up the word in the dictionary and add to word_indices if found
        # ====================== YOUR CODE HERE ======================
        for i, item in enumerate(vocabList):
            if word == item:
                word_indices.append(i+1)
        # =============================================================
    if verbose:
        print('----------------')
        print('Processed email:')
        print('----------------')
        print(' '.join(processed_email))
    return word_indices




world_indices = processEmail(email_contents)
print(world_indices)
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tensorflow as tf

data = pd.read_csv("ex1data1.txt",sep=",",header=None)
X = data.iloc[:,0:1]
y = data.iloc[:,1:]

X, y = np.array(X),np.array(y)

Weights = tf.Variable(tf.random_uniform([1]))
Bias = tf.Variable(tf.zeros([1]))

predict_y = Weights * X + Bias

loss = tf.reduce_mean(tf.square(y-predict_y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(2000):
    sess.run(train)
    if step%20 == 0:
        print(sess.run(Weights),sess.run(Bias))



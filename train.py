# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:57:24 2017

@author: lichunyang
"""

from readData import *
import tensorflow as tf
from scipy.misc import imread,imshow
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib.pyplot import plot
import numpy as np
import pandas as pd
images2 = readImages2('train-images.idx3-ubyte')
labels = readLabels('train-labels.idx1-ubyte')
labels = oneHot(labels)



print('Read Data Done.')
w = tf.Variable(tf.zeros([784,10],'float64'))
b = tf.Variable(tf.zeros([10],'float64'))

x = tf.placeholder('float64',[None,784])
y = tf.nn.softmax(tf.matmul(x,w)+b)
y_ = tf.placeholder('float64',[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(2000):
    batch_xs, batch_ys = getRandomData(images2,labels,100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if(i%50==0):
        print('iteration:%d' % i)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(sess.run(accuracy, feed_dict={x: images2, y_: labels}))

wv = (sess.run(w))
bv = (sess.run(w))
pd.DataFrame(wv).to_csv('softmax_W_t.csv')
pd.DataFrame(bv).to_csv('softmax_B_t.csv')

testImages = readImages2('t10k-images.idx3-ubyte')
testlabels = readLabels('t10k-labels.idx1-ubyte')
testlabels = oneHot(testlabels)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print('10K test:')
print(sess.run(accuracy, feed_dict={x: testImages, y_: testlabels}))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:52:21 2017

@author: HAL-9000
"""

import numpy as np
#import nems.keywords as nk
#import nems.utils as nu
#import nems.baphy_utils as bu
#import nems.modules as nm
#import nems.stack as ns
#import nems.fitters as nf
#import nems.main2 as mn
#import os
#import os.path
import copy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.matmul(x,W)+b
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(0,1500):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))ref    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
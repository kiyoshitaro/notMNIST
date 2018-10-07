# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:14:02 2018

@author: hungl
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import pandas as pd


pickle_file = 'notMNIST.pickle'

save = pd.read_pickle( 'notMNIST.pickle')

train_dataset = save['train_dataset']
train_labels = save['train_labels']
valid_dataset = save['valid_dataset']
valid_labels = save['valid_labels']
test_dataset = save['test_dataset']
test_labels = save['test_labels']
del save  # hint to help gc free up memory
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset,labels):
    dataset = dataset.reshape((dataset.shape[0],-1)).astype(np.float32)
    labels = [[(np.arange(num_labels) == label).astype(np.float32)] for label in labels]
    return np.array(dataset), np.array(labels).reshape((np.array(labels).shape[0],-1))

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


#BUILD GRAPH

train_subset = 10000
graph = tf.Graph()
with graph.as_default():
    #load data into constant to attach graph
    tf_train_dataset = tf.constant(train_dataset[:train_subset,:])
    tf_test_dataset = tf.constant(test_dataset)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_train_labels = tf.constant(train_labels[:train_subset])
    
    # Variables.
    weight = tf.Variable(tf.truncated_normal([image_size*image_size, num_labels])) #random values following a (truncated)

    biases = tf.Variable(tf.zeros([num_labels]))
    
    #computation
    logits = tf.matmul(tf_train_dataset,weight) + biases
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels,logits = logits))
    
    #Optimizer
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #Predict
    
    train_predict = tf.nn.softmax(logits)
    valid_predict = tf.nn.softmax(tf.matmul(tf_valid_dataset,weight)+biases)
    test_predict = tf.nn.softmax(tf.matmul(tf_test_dataset,weight)+biases)
    
#ITERATE:
num_steps = 800
def accuracy(predictions, labels):
    return(100.0*(np.sum(np.argmax(predictions,1) == np.argmax(labels,1))/predictions.shape[0]))

with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print("INIt")
    for step in range(num_steps):
        _,l,predictions = session.run([optimizer,loss,train_predict])
        if( step %100 ==0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels[:train_subset, :]))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
     print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    
    
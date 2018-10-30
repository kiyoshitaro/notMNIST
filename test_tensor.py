#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:54:56 2018

@author: kiyoshitaro
"""

import numpy as np
import tensorflow as tf
from datetime import datetime


#now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#root_logdir = "tf_logs"
logdir = "./graphs/normal_loading"


graph = tf.Graph()

with graph.as_default():
    a = tf.placeholder(dtype = tf.float32)
    b = tf.placeholder(dtype = tf.float32)
    c = tf.constant(5,dtype = tf.float32)
    sum_1 = tf.add(a,b)
    sum_2 = tf.add(b,c)
    res = sum_1*sum_2
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    
with tf.Session(graph = graph) as session:
    tf.initialize_all_variables().run()
    print("Init")
    feed_dict = {a : 3, b: 5}
    print(session.run(res,feed_dict = feed_dict))

file_writer.close()

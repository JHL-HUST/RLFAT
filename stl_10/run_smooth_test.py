"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time


import tensorflow as tf
import numpy as np
from collections import Counter

from model_bn import Model
import data_input

from utils import batch_brightness
np.random.seed(0)

with open('config.json') as config_file:
    config = json.load(config_file)

data_path = config['store_adv_path']

def run_attack(checkpoint, x_adv, epsilon):

  raw_data = data_input.Data(one_hot=True)

  x_input = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
  y_input = tf.placeholder(tf.int64, shape=[None, 10])

  model = Model(x_input, y_input, mode='eval')
  loss = model.xent
  grad = tf.gradients(loss, model.x_input)[0]
  sense = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(grad), -1), -1), -1)))
  print(sense.shape)

  saver = tf.train.Saver()

  num_eval_examples = x_adv.shape[0]
  eval_batch_size = 100

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0

  x_nat = raw_data.eval_data.xs
  l_inf = np.amax(np.abs(x_nat - x_adv))

  if l_inf > epsilon + 0.0001:
    print('maximum perturbation found: {}'.format(l_inf))
    print('maximum perturbation allowed: {}'.format(epsilon))
    #return

  y_pred = [] # label accumulator

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)
    sense_val = 0
    # Iterate over the samples batch-by-batch
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = x_adv[bstart:bend, :]
      y_batch = raw_data.eval_data.ys[bstart:bend]
      
      x_batch = batch_brightness(x_batch, c=-0.15)
      
      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch}
      
      sense_val += sess.run(sense,  feed_dict=dict_adv)
      
      
    print(sense_val/(num_eval_examples)) 
  
  
  #print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
  #y_pred = np.concatenate(y_pred, axis=0)
  #print(y_pred.shape)
  #print(Counter(y_pred))
  '''
  y_pred = np.concatenate(y_pred, axis=0)
  np.save('pred.npy', y_pred)
  print('Output saved at pred.npy')
  '''

if __name__ == '__main__':

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_dir = config['model_dir']

  checkpoint = tf.train.latest_checkpoint(model_dir)
  x_adv = np.load(config['store_adv_path'])

  print(checkpoint+"  "+config['store_adv_path'])

  if checkpoint is None:
    print('No checkpoint found')
  elif x_adv.shape != (8000, 96, 96, 3):
    print('Invalid shape: expected (8000, 96, 96, 3), found {}'.format(x_adv.shape))
  elif np.amax(x_adv) > 1.0001 or np.amin(x_adv) < -0.0001:
    print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
                                                              np.amin(x_adv),
                                                              np.amax(x_adv)))
  else:
    run_attack(checkpoint, x_adv, config['epsilon'])

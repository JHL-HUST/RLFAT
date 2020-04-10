"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import data_input

from model_bn import Model
from pgd_attack import LinfPGDAttack
from utils import rbs_transformation

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
raw_data = data_input.Data(one_hot=True)
global_step = tf.contrib.framework.get_or_create_global_step()

s_x_input = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
s_y_input = tf.placeholder(tf.float32, shape=[None, 10])
s_model = Model(s_x_input, s_y_input, mode='train')

u_x_input = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
u_y_input = tf.placeholder(tf.float32, shape=[None, 10])
u_model = Model(u_x_input, u_y_input, mode='train', reuse=True)

# Setting up the optimizer
lambda_num = 1.0
loss_ft = tf.reduce_sum(tf.squared_difference(u_model.pre_softmax, s_model.pre_softmax)) * 1.0 / batch_size
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(u_model.mean_xent + 0.5 * loss_ft,
                                                       global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(s_model,
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

saver = tf.train.Saver(max_to_keep=3)
LATEST_CHECKPOINT = tf.train.latest_checkpoint(model_dir)
tf.summary.scalar('accuracy adv train', s_model.accuracy)
tf.summary.scalar('xent adv train', s_model.xent / batch_size)
tf.summary.image('images adv train', s_model.x_input)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

  data_flow = data_input.AugmentedData(raw_data, sess)
  start_step = 0
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  if LATEST_CHECKPOINT:
    print("Restore session from checkpoint: {}".format(LATEST_CHECKPOINT))
    saver.restore(sess, LATEST_CHECKPOINT)
    start_step = int(LATEST_CHECKPOINT.split('-')[-1])

  training_time = 0.0
  

  # Main training loop
  for ii in range(start_step, max_num_training_steps+1):
    x_batch, y_batch = data_flow.train_data.get_next_batch(batch_size, multiple_passes=True)

    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    end = timer()
    training_time += end - start

    x_batch_s = rbs_transformation(x_batch_adv, 2)

    mixed_y = y_batch
    mixed_x_adv = x_batch_s

    nat_dict = {s_model.x_input: x_batch,
                s_model.y_input: y_batch}

    adv_dict = {s_model.x_input: x_batch_adv,
                s_model.y_input: y_batch,
                u_model.x_input: mixed_x_adv,
                u_model.y_input: mixed_y}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(s_model.accuracy, feed_dict=adv_dict)
      adv_acc = sess.run(u_model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training adv accuracy {:.4}%'.format(nat_acc * 100))
      print('    training s adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start

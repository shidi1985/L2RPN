__author__ = 'tulan'
# Copyright (C) 2018 - 2019 GEIRI North America
# Authors: Tu Lan <tu.lan@geirina.net>

import os
import tensorflow as tf
import numpy as np
import pypownet.environment
from abc import ABC, abstractmethod


class Agent(ABC):
  def __init__(self, environment):
    """Initialize a new agent."""
    assert isinstance(environment, pypownet.environment.RunEnv)
    self.environment = environment

  @abstractmethod
  def act(self, observation):
    pass

  def feed_reward(self, action, consequent_observation, rewards_aslist):
    pass


class DeepQNetworkDueling(Agent):
  def __init__(self,
               environment,
               n_state=538,
               n_actions=251,
               learning_rate=1e-4,
               scope='dqn',
               summaries_dir=None):
    """
    Deep Q Network with experience replay and fixed Q-target

    Args:
      environment: env
      learning_rate: model learning rate
      scope: graph scope name
      summaries_dir: log directory of Tensorboard Filewriter

    Returns:
      pass
    """

    super().__init__(environment)
    self.learning_rate = learning_rate
    self.n_state = n_state
    self.n_actions = n_actions
    self.scope = scope
    self.summary_writer = None

    # Build model and create Tensorboard summary
    with tf.variable_scope(scope):
      self._build_model()
      if summaries_dir:
        summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
        if not os.path.exists(summary_dir):
          os.makedirs(summary_dir)
        self.summary_writer = tf.summary.FileWriter(summary_dir)

  def _build_model(self):
    """
    Build Tensorflow graph
    """
    with tf.variable_scope('inputs'):
      # State as Vector
      self.s = tf.placeholder(
          shape=[None, self.n_state], dtype=tf.float32, name='s')
      # Placeholder for TD target
      self.q_target = tf.placeholder(
          shape=[None], dtype=tf.float32, name='q_target')
      # Placeholder for actions
      self.action_ph = tf.placeholder(
          shape=[None, 1], dtype=tf.int32, name='action')
      # Placeholder for Importance-Sampling(IS) weights
      self.IS_weights = tf.placeholder(
          shape=[None], dtype=tf.float32, name='IS_weights')
      # imitation learning label
      self.label = tf.placeholder(
          shape=[None, self.n_actions], dtype=tf.float32, name='label')

    # forward pass
    def forward_pass(s):
      # Only FC Layer, for vector input
      # fc1 common layer
      s = tf.layers.batch_normalization(s, name='bn')

      fc1 = tf.contrib.layers.fully_connected(
          s, 256, activation_fn=tf.nn.relu, scope='fc1')

      # fc_V
      fc_V = tf.contrib.layers.fully_connected(
          fc1, 32, activation_fn=tf.nn.relu, scope='fc_V_1')
      self.fc_V = tf.contrib.layers.fully_connected(
          fc_V, 1, activation_fn=None, scope='fc_V_2')

      # fc_A
      fc_A = tf.contrib.layers.fully_connected(
          fc1, 128, activation_fn=tf.nn.relu, scope='fc_A_1')
      self.fc_A = tf.contrib.layers.fully_connected(
          fc_A, self.n_actions, activation_fn=None, scope='fc_A_2')

      with tf.variable_scope('q_predict'):
        mean_A = tf.reduce_mean(self.fc_A, axis=1, keep_dims=True)
        q_predict = tf.add(self.fc_V, tf.subtract(self.fc_A, mean_A))
        # q_predict = tf.tanh(q_predict)

      return q_predict

    # forward pass
    self.q_predict = forward_pass(self.s)

    # q_predict for chosen action
    indexes = tf.reshape(tf.range(tf.shape(self.q_predict)[0]), shape=[-1, 1])
    action_indexes = tf.concat([indexes, self.action_ph], axis=1)
    self.q_predict_action = tf.gather_nd(self.q_predict, action_indexes)

    # calculate the mean batch loss
    self.abs_TD_errors = tf.abs(self.q_target - self.q_predict_action)
    self.loss_batch = self.IS_weights * \
        tf.squared_difference(self.q_target, self.q_predict_action)

    with tf.variable_scope('loss'):
      self.loss = tf.reduce_mean(self.loss_batch)

    # optimizer
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    # train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.variable_scope('train_op'):
      with tf.control_dependencies(update_ops):
        self.train_op = self.optimizer.minimize(self.loss, tf.train.get_or_create_global_step())

    best_index = tf.reshape(tf.argmax(self.label, axis=-1, output_type=tf.dtypes.int32), shape=[-1, 1])
    best_indexes = tf.concat([indexes, best_index], axis=1)
    q_predict_best = tf.gather_nd(self.q_predict, best_indexes)
    label_best = tf.gather_nd(self.label, best_indexes)
    imit_loss_main = tf.reduce_mean(tf.squared_difference(q_predict_best, label_best))
    imit_loss_other = tf.reduce_mean(tf.squared_difference(self.q_predict, self.label))

    # label_max = tf.reduce_max(self.label)
    # predict_max = tf.gather_nd(self.q_predict, index_max)

    optimizer_imit = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    with tf.variable_scope('imitation_learning'):
      with tf.control_dependencies(update_ops):
        self.imit_loss = imit_loss_main * 0.7 + imit_loss_other * 0.3
        # self.imit_loss = imit_loss_other
        self.imit_loss = tf.losses.huber_loss(self.label, self.q_predict)
        self.train_op_imit = optimizer_imit.minimize(self.imit_loss, tf.train.get_or_create_global_step())

    # trainable parameters
    self.params_train = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    # Summaries for Tensorboard
    # self.summaries = tf.summary.merge([
    #     tf.summary.histogram("loss_hist", self.loss_batch),
    #     tf.summary.histogram("q_values_hist", self.q_predict),
    # ])

  def epsilon_greedy(self, q_predict, epsilon):
    roll = np.random.uniform()
    if roll < epsilon:
      return np.random.randint(self.n_actions)
    else:
      return np.argmax(q_predict[0])

  def act(self, sess, s, epsilon):
    q_predict = sess.run(self.q_predict, feed_dict={self.s: s})
    return self.epsilon_greedy(q_predict, epsilon), q_predict[0]

  def predict(self, sess, s):
    return sess.run([self.q_predict], feed_dict={self.s: s})

  def update(self, sess, s, q_target, action, IS_weights):
    feed_dict = {self.s: s, self.q_target: q_target,
                 self.action_ph: action, self.IS_weights: IS_weights}
    _, loss, abs_TD_errors = sess.run(
        [self.train_op, self.loss, self.abs_TD_errors], feed_dict)
    # if self.summary_writer:
    #   self.summary_writer.add_summary(summaries)
    return loss, abs_TD_errors
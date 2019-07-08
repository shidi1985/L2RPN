
import pypownet.agent
import pypownet.environment
import tensorflow as tf

import sys

import pickle
import os

import numpy as np
import time


class DeepQNetworkDueling():
  def __init__(self,
               environment,
               n_state=538,
               n_actions=1000,
               learning_rate=1e-3,
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

    self.environment = environment
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


  def act(self, sess, s):
      q_predict = sess.run(self.q_predict, feed_dict={self.s: s})
      return np.argmax(q_predict[0]), q_predict[0]




class Submission(pypownet.agent.Agent):

    def __init__(self,
             environment,
             render=False,
             verbose=False,
             vverbose=False,
             parameters=None,
             level=None,
             max_iter=None,
             log_filepath='runner.log',
             machinelog_filepath='machine_logs.csv',
             n_features=538,
             n_episode=10000,
             learning_rate=0.01,
             gamma=0.99,
             replace_target_iter=300,
             replay_memory_size=300,
             PER_alpha=0.6,
             PER_beta=0.4,
             batch_size=32,
             epsilon_start=1.0,
             epsilon_end=0.1,
             epsilon_decay_steps=1000,
             verbose_per_episode=100,
             seed=22,
             worker_id=0):

        # Sanity checks: both environment and agent should inherit resp. RunEnv and Agent
        super().__init__(environment)

        # training params
        self.environment = environment
        self.n_features = n_features
        self.n_episode = n_episode
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.replay_memory_size = replay_memory_size
        self.PER_alpha = PER_alpha
        self.PER_beta = PER_beta
        self.batch_size = batch_size
        self.verbose_per_episode = verbose_per_episode
        self.seed = seed
        self.worker_id = worker_id
        
        # state control
        self.seed = seed
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # control epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = 3 * n_episode // 4

        # Directory
        self.data_dir = os.path.abspath('example_submission')
        action_dir = os.path.join(self.data_dir, 'actions_251.npy')

        # load actions
        self.action_space = np.load(action_dir, allow_pickle=True)
        self.n_actions = 251
        self.action_count = {}

        # build graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.dqn_main = DeepQNetworkDueling(environment, learning_rate=learning_rate, n_actions=self.n_actions,
                                              scope='dqn_main')
            self.dqn_target = DeepQNetworkDueling(environment, learning_rate=learning_rate, n_actions=self.n_actions,
                                                scope='dqn_target')
            # Saver
            self.saver = tf.train.Saver()

        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.sess = tf.Session(config=config, graph=self.graph)
        # self.sess.run(self.initializer)

        # load model
        model_name = '2_FC_model_251_imitation_40000_batch_1'
        with self.sess.as_default():
            self.saver.restore(self.sess, self.data_dir + '/{}.ckpt'.format(model_name))
            print('Model Loaded!')


    def get_lineflow(self, obs):
      p_origin = np.array(obs[256: 276])
      q_origin = np.array(obs[276: 296])
      v_origin = np.array(obs[296: 316])

      # print(p_origin, '\n', q_origin)
      s_origin = 1000. * np.divide(np.sqrt(np.power(p_origin, 2) + np.power(q_origin, 2)), (3 ** 0.5 * v_origin * 100))
      lineflow = [round(x, 4) for x in s_origin]
      return lineflow


    # take a step
    def act(self, observation):
        time_start = time.time()
        lineflow = self.get_lineflow(observation)

        observation = np.array(observation).reshape((-1, self.n_features))
        # print(self.environment.game.__chronic_looper.get_current_chronic_name())
        # print('P load: ', observation[0][:11])
        # choose action
        action, q_predictions = self.dqn_main.act(self.sess, observation)
        # ...
        action = 0

        # check loadflow
        grid = self.environment.game.grid
        thermal_limits = grid.get_thermal_limits()
        current_flows_a = grid.extract_flows_a()
        print('Cur lineflow: ', np.round(current_flows_a / thermal_limits, 3).tolist())

        # check overflow
        has_overflow = self.environment.game.n_timesteps_soft_overflowed_lines
        print('overflow lines: ', has_overflow)
        has_overflow = any(has_overflow)
        action_is_valid = True
        has_danger = False

        # if the action will terminate, try others
        obs_simulate, reward_simulate_pre, done_simulate, _, score_simulate_pre= self.environment.simulate(
              self.action_space[action])


        if obs_simulate is None:
          has_danger = True
        else:
          lineflow_simulate = self.get_lineflow(obs_simulate)
          lineflow_simulate_ratio = lineflow_simulate / thermal_limits
          lineflow_simulate_ratio = [round(x, 4) for x in lineflow_simulate_ratio]
          if any([x > 0.93 for x in lineflow_simulate_ratio]):
            has_danger = True
          print('lineflow: ', lineflow_simulate_ratio)

        print(action, score_simulate_pre, done_simulate, action_is_valid)


        print(self.environment.game.current_timestep_id)
        if (done_simulate or not action_is_valid or score_simulate_pre < 13.50 or has_overflow or has_danger) and self.environment.game.current_timestep_id >= 215:
          # if has overflow. try all actions
          if has_overflow:
            print('has overflow !!!!!!!!!!!!!!!!!!!!!!!!!')
          if has_danger:
            print('has danger !!!!!!!!!!!!!!!!!!!!!!!!!!!')

          top_actions = np.argsort(q_predictions)[-1: -50: -1].tolist()

          chosen_action = 0
          max_score = float('-inf')
          for action in top_actions:
            action_class = self.environment.action_space.array_to_action(self.action_space[action])
            action_is_valid = self.environment.game.is_action_valid(action_class)
            if not action_is_valid:
              continue
            else:
              obs_simulate, reward_simulate, done_simulate, _, score_simulate= self.environment.simulate(
                self.action_space[action])

            if obs_simulate is None:
              continue
            else:
              lineflow_simulate = self.get_lineflow(obs_simulate)
              lineflow_simulate_ratio = lineflow_simulate / thermal_limits
              lineflow_simulate_ratio = [round(x, 4) for x in lineflow_simulate_ratio]

              # has_danger = any([x > 0.92 for x in lineflow_simulate_ratio])
              # seperate big line and small line
              has_danger = False
              if any([x > 0.93 for x in lineflow_simulate_ratio]):
                has_danger = True

              if not done_simulate and score_simulate > max_score and not has_danger:
                max_score = score_simulate
                chosen_action = action
                print('lineflow: ', lineflow_simulate_ratio)
                print('current best action: {}, score: {:.4f}'.format(chosen_action, score_simulate))

          # chosen action
          action = chosen_action

        time_end = time.time()
        print('chosen_action: {}, time_used: {:.4f}'.format(action, time_end - time_start))
        sys.stdout.flush()

        self.action_count[action] = self.action_count.get(action, 0) + 1
        print(self.action_space[action])
        return self.action_space[action]

#if you want to load a file (in this directory) names "model.dupm"
#open("program/model.dump"
__author__ = 'tulan'
# Copyright (C) 2018 - 2019 GEIRI North America
# Authors: Tu Lan <tu.lan@geirina.net>, Jiajun Duan <jiajun.duan@geirina.net>, Bei Zhang <bei.zhang@geirina.net>

from pypownet.prioritized_memory import Memory
from pypownet.agent import DeepQNetworkDueling
import time
import os
import random
import itertools
import collections
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime

from collections import defaultdict, deque
from pypownet.environment import RunEnv
from pypownet.agent import Agent


class Runner(object):
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
    assert isinstance(environment, RunEnv)

    # training params
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
    self.data_dir = os.path.abspath('data')
    action_to_index_dir = os.path.join(self.data_dir, 'action_to_index.npy')
    action_dir = os.path.join(self.data_dir, 'actions_176.npy')
    log_dir = os.path.abspath("./logs")
    self.model_dir = os.path.join(self.data_dir, 'model_saved')
    self.results_dir = os.path.abspath("./results")
    if not os.path.exists(self.results_dir):
      os.mkdir(self.results_dir)

    # load actions
    self.action_space = np.load(action_dir, allow_pickle=True)
    self.n_actions = self.action_space.shape[0]
    self.action_to_index = np.load(action_to_index_dir, allow_pickle=True)

    # build graph
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.dqn_main = DeepQNetworkDueling(environment, learning_rate=learning_rate, n_actions=self.n_actions,
                                          scope='dqn_main')
      self.dqn_target = DeepQNetworkDueling(environment, learning_rate=learning_rate, n_actions=self.n_actions,
                                            scope='dqn_target')
      # Saver
      self.saver = tf.train.Saver()

    # The replay memory
    self.replay_memory = Memory(replay_memory_size, PER_alpha, PER_beta)

    # hard copy
    self.params_copy_hard = [target.assign(main) for main, target in zip(
        self.dqn_main.params_train, self.dqn_target.params_train)]

    # summary
    self.timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
    self.episode_reward_history = [0] * self.n_episode
    self.episode_score_history = [0] * self.n_episode
    self.episode_tot_history = [0] * self.n_episode

    # good actions
    self.actions_good = [20, 160, 60, 100, 80, 40, 140, 165, 120, 170, 105, 59, 19, 5, 3]

    self.environment = environment
    self.verbose = verbose
    self.render = render

    if self.render:
      self.environment.render()

  def get_statemap(self, observation):

    def map_sub_gen_load(sub_id, switch_id, p, n_sub):
      tmp1 = np.zeros(n_sub)
      original = switch_id == 0
      sub_id_original = sub_id[original]
      p1 = p[original]
      tmp1[sub_id_original - 1] = p1

      tmp2 = np.zeros(n_sub)
      split = switch_id == 1
      sub_id_split = sub_id[split]
      p2 = p[split]
      tmp2[sub_id_split - 1] = p2

      tmp = np.concatenate((tmp1, tmp2), axis = None)

      return copy.deepcopy(tmp)

    base = 100.0
    n_sub = 14

    load_p = observation[0:11]
    load_is_off = observation[11:22].astype(np.int32)
    load_sub_id = observation[406:417].astype(np.int32)
    load_switch_id = observation[33:44].astype(np.int32)
    load_p_on = load_p * (1 - load_is_off)
    load_map = map_sub_gen_load(load_sub_id, load_switch_id, load_p_on, n_sub)

    gen_p = observation[44:49]
    gen_is_off = observation[49:54].astype(np.int32)
    gen_sub_id = observation[417:422].astype(np.int32)
    gen_switch_id = observation[59:64].astype(np.int32)
    gen_p_on = gen_p * (1 -gen_is_off)
    gen_map = map_sub_gen_load(gen_sub_id, gen_switch_id, gen_p_on, n_sub)

    net_injection_diagonal = (gen_map - load_map) / base

    line_ampre = observation[104:124]
    line_cap = observation[462:482]
    line_is_on = observation[124:144].astype(np.int32)
    line_from_sub_id = observation[422:442].astype(np.int32)
    line_to_sub_id = observation[442:462].astype(np.int32)
    line_from_switch_id = observation[64:84].astype(np.int32)
    line_to_switch_id = observation[84:104].astype(np.int32)

    line_flow_ratio = line_ampre * line_is_on / line_cap

    line_from_switch = line_from_switch_id == 1
    line_from_sub_id[line_from_switch] += n_sub
    line_to_switch = line_to_switch_id == 1
    line_to_sub_id[line_to_switch] += n_sub

    obs_matrix = np.zeros([n_sub * 2, n_sub * 2])
    obs_matrix[line_from_sub_id - 1 , line_to_sub_id - 1] = line_flow_ratio
    obs_matrix[line_to_sub_id - 1, line_from_sub_id -1] = line_flow_ratio
    obs_matrix[range(0, n_sub * 2), range(0, n_sub * 2)] = net_injection_diagonal

    return copy.deepcopy(obs_matrix)


  def get_score(self):
    score = self.environment.reward_signal.get_score(self.environment.game.grid)
    return score

  def get_lineflow(self, obs):
    p_origin = np.array(obs[256: 276])
    q_origin = np.array(obs[276: 296])
    p_ex = np.array(obs[316: 336])
    q_ex = np.array(obs[336: 356])

    s_origin = np.sqrt(np.power(p_origin, 2), np.power(q_origin, 2))
    s_ex = np.sqrt(np.power(p_ex, 2), np.power(q_ex, 2))
    lineflow = np.maximum(s_origin, s_ex)
    return lineflow

  def test(self, model_name='', episode_per_step=1000):
    time_start = time.time()
    score_sum = 0.0
    # session config
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)

    with tf.Session(config=config, graph=self.graph) as sess:
      sess.run(tf.global_variables_initializer())

      # load model
      self.saver.restore(sess, self.model_dir + '/{}.ckpt'.format(model_name))
      print('Model {} Loaded!'.format(model_name))

      step_tot = 0
      loss = 0


      print("Start testing...")

      done = False
      flag_next_chronic = False
      # self.environment.game.current_timestep_id = 7450

      for episode in range(self.n_episode):
        # if done or flag_next_chronic:
        state = self.environment.reset() if episode != 0 else self.environment._get_obs().as_array()
        state = np.array(state).reshape((-1, self.n_features))
        count_action = {}

        episode_time_start = time.time()

        for step in itertools.count():
          # print invalid action
          print('reconnect: {}\n reaction_line: {}, reaction_node: {}'. format(
            self.environment.game.timesteps_before_lines_reconnectable, 
            self.environment.game.timesteps_before_lines_reactionable,
            self.environment.game.timesteps_before_nodes_reactionable))

          time_start_step = time.time()
          # choose action
          action, q_predictions= self.dqn_main.act(sess, state, 0)

          # if the action will terminate, try others
          obs_simulate, reward_simulate, done_simulate, _, score_simulate_pre = self.environment.simulate(
                self.action_space[action])
          action_class = self.environment.action_space.array_to_action(self.action_space[action])
          action_is_valid = self.environment.game.is_action_valid(action_class)

          print(score_simulate_pre)

          if done_simulate or not action_is_valid or score_simulate_pre < 14.00:

            top_action_100 = np.argsort(q_predictions)
            chosen_action = 20
            score_simulate = -1
            i = 1
            while (done_simulate or score_simulate <= score_simulate_pre) and i < 20:
              action = top_action_100[i]
              i += 1
              action_class = self.environment.action_space.array_to_action(self.action_space[action])
              action_is_valid = self.environment.game.is_action_valid(action_class)
              if not action_is_valid:
                continue

              obs_simulate, reward_simulate, done_simulate, _, score_simulate = self.environment.simulate(
                  self.action_space[action])

              if not done_simulate and score_simulate > score_simulate_pre:
                chosen_action = action
              print(action, score_simulate, done_simulate)

            # count simulate
            action = chosen_action

          # count action
          count_action[action] = count_action.get(action, 0) + 1

          next_state, reward, done, info, flag_next_chronic = self.environment.step(
              self.action_space[action])

          # calc and normalize total reward
          score = self.get_score()
          reward_tot = score

          # record
          self.episode_score_history[episode] += score
          score_sum += score

          if done:
            next_state = state
          else:
            next_state = np.array(next_state).reshape((-1, self.n_features))

          # verbose step summary
          if episode % self.verbose_per_episode == 0 and (step_tot + 1) % 1 == 0:
            print("\repisode: {}, step: {}, action: {}, reward: {:4f}, score: {:.4f}, tot: {:.4f}".
                  format(episode + 1, step + 1, action, reward, score, reward_tot))

            sys.stdout.flush()

          state = next_state
          step_tot += 1

          if done or step >= episode_per_step - 1:
            break

        episode_time_end = time.time()
        # verbose episode summary
        print("\nepisode: {}, mean_score: {:4f}, sum_score: {:4f}, time_used: {:.4f}\n".
                  format(episode + 1, self.episode_score_history[episode] / (step + 1), 
                    self.episode_score_history[episode], episode_time_end - episode_time_start))
        print(count_action)


      time_end = time.time()
      print('\nSum Reward of 10 episodes: {:.4f}, Mean Score: {:.4f}'.format(score_sum, score_sum / 10.0))
      print("\nFinished, Total time used: {}s".format(time_end - time_start))

  # generate labels
  def generate_rewards(self, max_step=1000):

    reward_dir = os.path.join(self.data_dir, 'state_action_reward_3120')
    score_dir = os.path.join(self.data_dir, 'state_action_score_3120/160_199')
    tot_dir = os.path.join(self.data_dir, 'state_action_tot_3120')

    if not os.path.exists(reward_dir):
      os.mkdir(reward_dir)
    if not os.path.exists(score_dir):
      os.mkdir(score_dir)
    if not os.path.exists(tot_dir):
      os.mkdir(tot_dir)

    time_start = time.time()
    state_action_reward = defaultdict(list)
    state_action_score = defaultdict(list)
    state_action_tot = defaultdict(list)
    # session config
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)

    with tf.Session(config=config, graph=self.graph) as sess:
      sess.run(tf.global_variables_initializer())
      step_tot = 0

      grid = self.environment.game.grid
      thermal_limits = grid.get_thermal_limits()

      print("Start Running...")

      for episode in range(self.n_episode):
        state = self.environment.reset() if episode != 0 else self.environment._get_obs().as_array()
        state = np.array(state).reshape((-1, self.n_features))

        for step in itertools.count():
          # update the target estimator
          max_reward = float('-inf')
          max_score = float('-inf')
          max_tot = float('-inf')
          chosen_action = 0
          for i, action in enumerate(self.action_space):
            # verify action
            is_valid = self.environment.game.is_action_valid(self.environment.action_space.array_to_action(action))
            if not is_valid:
              reward, score, tot = 0, -10, -10
            else:
              obs, reward, done, _, score = self.environment.simulate(action)

              if done or score == 0:
                reward, score, tot = 0, -10, -10
              else:
                lineflow_simulate = self.get_lineflow(obs)
                lineflow_simulate_ratio = lineflow_simulate / thermal_limits
                lineflow_simulate_ratio = [round(x, 4) for x in lineflow_simulate_ratio]

                if any([x >= 0.99 for x in lineflow_simulate_ratio]):
                  reward, score, tot = 0, -10, -10
            
            tot = (reward + score) / 10.0

            state_tuple = tuple(state[0])
            state_action_reward[state_tuple].append(reward)
            state_action_score[state_tuple].append(score)
            state_action_tot[state_tuple].append(tot)

            if reward > max_reward:
              max_reward = reward
            if score > max_score:
              max_score = score
              chosen_action = i
            if tot > max_tot:
              max_tot = tot

            if (i + 1) % 50 == 0:
              print("\rworker: {}, episode: {}, step: {},  ith_action: {}, max_score: {:.4f}, max_tot: {:.4f}".
                    format(self.worker_id, episode + 1, step + 1, i + 1, max_score, max_tot))
              sys.stdout.flush()


          next_state, reward, done, _, _ = self.environment.step(self.action_space[chosen_action])

          if not done:
            next_state = np.array(next_state).reshape((-1, self.n_features))

          state = next_state
          step_tot += 1

          if max_score == 0:
            print('worker {} has dead!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(self.worker_id))
          # print every step
          print("\rworker: {}, episode: {}, step: {}, max_score: {:.5f}, max_tot: {:.5f}\naction:\n{}".
                format(self.worker_id, episode + 1, step + 1, max_score, max_tot, chosen_action))
          sys.stdout.flush()

          if step_tot % 10 == 0:
            # np.save(os.path.join(reward_dir, 'worker_{}_step_{}.npy'.format(self.worker_id, max_step)),
            #   state_action_reward)
            np.save(os.path.join(score_dir, 'worker_{}_step_{}.npy'.format(self.worker_id, max_step)),
                    state_action_score)
            # np.save(os.path.join(tot_dir, 'worker_{}_step_{}.npy'.format(self.worker_id, max_step)),
            #         state_action_tot)
            print("\nReward and Score saved!") 

          if done or step_tot >= max_step:
            break

        if step_tot >= max_step:
          break

      time_end = time.time()
      print("\nFinished, Total time used: {}s".format(time_end - time_start))

      # save the generated reward matrix
      # np.save(os.path.join(reward_dir, 'worker_{}_step_{}.npy'.format(self.worker_id, max_step)),
      #         state_action_reward)
      np.save(os.path.join(score_dir, 'worker_{}_step_{}.npy'.format(self.worker_id, max_step)),
              state_action_score)
      # np.save(os.path.join(tot_dir, 'worker_{}_step_{}.npy'.format(self.worker_id, max_step)),
      #         state_action_tot)
      print("\nReward and Score saved!") 


  def train_imitation(self, data_size=19720, hasVal=True, test=False, model_name=None):
    X_train = np.load(self.data_dir + '/X_train_{}.npy'.format(data_size), allow_pickle=True)
    X_val = np.load(self.data_dir + '/X_val_{}.npy'.format(data_size), allow_pickle=True)
    y_train = np.load(self.data_dir + '/y_train_{}.npy'.format(data_size), allow_pickle=True)
    y_val = np.load(self.data_dir + '/y_val_{}.npy'.format(data_size), allow_pickle=True)
    print('data loaded')
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    time_start = time.time()
    num_iter_per_epoch = X_train.shape[0] // self.batch_size

    # session config
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    print('Strat Training ...')
    with tf.Session(config=config, graph=self.graph) as sess:
      sess.run(tf.global_variables_initializer())
      plt.ion()
      plt.show()
      # load model
      if test:
        self.saver.restore(sess, self.model_dir + '/model_3120_imitation_45125.ckpt')
        print('Model Loaded!')

      if model_name:
        self.saver.restore(sess, self.model_dir + '/{}.ckpt'.format(model_name))
        print('Model Loaded!')        

      # Training cycle
      for episode in range(self.n_episode):
        start_index = 0
        # Loop over all batches
        for i in range(num_iter_per_epoch):
          if i == num_iter_per_epoch - 1:
            X_bacth = X_train[start_index:]
            y_batch = y_train[start_index:]
          else:    
            X_bacth = X_train[start_index:start_index + self.batch_size]
            y_batch = y_train[start_index:start_index + self.batch_size]
          start_index += self.batch_size

          # Run optimization op (backprop) and cost op (to get loss value)
          if not test:
            # q_best, label_best = sess.run([self.dqn_main.q_predict_best, self.dqn_main.label_best],
            #   feed_dict={self.dqn_main.s: X_bacth, self.dqn_main.label: y_batch})
            # print(q_best, label_best)

            _, train_loss = sess.run([self.dqn_main.train_op_imit, self.dqn_main.imit_loss], 
                feed_dict={self.dqn_main.s: X_bacth, self.dqn_main.label: y_batch})
            if hasVal:
              random_s = np.random.randint(y_val.shape[0] - 11)
              val_loss = sess.run(self.dqn_main.imit_loss, feed_dict={
                self.dqn_main.s: X_val[random_s: random_s + 10], self.dqn_main.label: y_val[random_s: random_s + 10]})

            action, q_predictions = self.dqn_main.act(sess, X_bacth[0].reshape((1, self.n_features)), 0)
            print('y: {}\nq_predict: {}, mean_q: {:.5f}\n'.format(np.argmax(y_batch[0]), action, np.mean(q_predictions)))
            print('episode: {}, train_loss: {:.5f}, val_loss: {:.5f}\n'.format(episode + 1, train_loss, val_loss))

          else:
            action, q_predictions = self.dqn_main.act(sess, X_bacth[0].reshape((1, self.n_features)), 0)
            print('y: {}\nq_predict: {}\n'.format(np.argmax(y_batch[0]), action))

          # save model
          # if (i + 1) % 2000 == 0 and not test:
            # print('Model Saved!')
            # self.saver.save(sess, self.model_dir + '/1_FC_model_251_imitation_{}_batch_{}.ckpt'.format(data_size, self.batch_size))
          # plot
          if (i + 1) % 100  == 0 and not test:
            # plot sample
            plt.clf()
            plt.plot(range(1, 177), q_predictions, 'g-', label='Prediction')
            plt.plot(range(1, 177), y_batch[0], 'b--', label='Label')
            plt.xlabel('Action')
            plt.ylabel('Score')
            plt.legend()
            plt.title('Imitation Learning Prediction & Label')
            plt.draw()
            plt.pause(0.001)
            plt.savefig(self.data_dir + '{}_imit.png'.format(i))

        # control verbose
        if (episode + 1) % 1 == 0 and hasVal and not test:
          print('episode: {}, train_loss: {:.5f}, val_loss: {:.5f}\n'.format(episode + 1, train_loss, val_loss))
        

      # hard copy to target network
      sess.run([self.params_copy_hard])

      # save model
      self.saver.save(sess, self.model_dir + '/1_FC_model_251_imitation_{}_batch_{}.ckpt'.format(data_size, self.batch_size))
      print('Model Saved!')

      time_end = time.time()
      print("\nFinished, Total time used: {}s".format(time_end - time_start))


  def train(self, total_train_step=100000, model_name=None, isco=''):

    time_start = time.time()
    epsilons = np.linspace(
        self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps)
    count_simulate = [0] * self.n_episode
    count_action = {}

    # session config
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)

    with tf.Session(config=config, graph=self.graph) as sess:
      sess.run(tf.global_variables_initializer())
      # self.dqn_main.summary_writer.add_graph(sess.graph)
      # load model
      if model_name:
        self.saver.restore(sess, self.model_dir + '/{}.ckpt'.format(model_name))
        print('Model {} Loaded!'.format(model_name))

      step_tot = 0
      loss = 0

      print("Start training...")
      sys.stdout.flush()

      done = False
      flag_next_chronic = False

      for episode in range(self.n_episode):
        # if done or flag_next_chronic:
        state = self.environment.reset() if episode != 0 else self.environment._get_obs().as_array()
        state = np.array(state).reshape((-1, self.n_features))

        for step in itertools.count():
          reward_tot = 0
          # update the target estimator
          if step_tot % self.replace_target_iter == 0:
            sess.run([self.params_copy_hard])
          # print("\nCopied model parameters to target network.")

          # choose action
          action, q_predictions = self.dqn_main.act(
              sess, state, epsilons[min(episode, self.epsilon_decay_steps - 1)])

          # check loadflow
          grid = self.environment.game.grid
          thermal_limits = grid.get_thermal_limits()

          # check overflow
          has_overflow = self.environment.game.n_timesteps_soft_overflowed_lines
          print('overflow lines: ', has_overflow)
          has_overflow = any(has_overflow)
          action_is_valid = True
          has_danger = False

          # if the action will terminate, try others
          obs_simulate, reward_simulate, done_simulate, _, score_simulate_pre= self.environment.simulate(
                self.action_space[action])

          if obs_simulate is None:
            has_danger = True
          else:
            lineflow_simulate = self.get_lineflow(obs_simulate)
            lineflow_simulate_ratio = lineflow_simulate / thermal_limits
            lineflow_simulate_ratio = [round(x, 4) for x in lineflow_simulate_ratio]
            for ratio, limit in zip(lineflow_simulate_ratio, thermal_limits):
              if (limit < 400.00 and ratio > 0.90) or (limit >= 400.00 and ratio > 0.95):
                has_danger = True
            print('lineflow: ', lineflow_simulate_ratio)

          print(action, score_simulate_pre, done_simulate, action_is_valid)

          if done_simulate or not action_is_valid or score_simulate_pre < 13.50 or has_overflow or has_danger:
            # if has overflow. try all actions
            if has_overflow:
              print('has overflow !!!!!!!!!!!!!!!!!!!!!!!!!')
            if has_danger:
              print('has danger !!!!!!!!!!!!!!!!!!!!!!!!!!!')

            top_actions = np.argsort(q_predictions)[-1: -41: -1].tolist()
            top_actions = set(top_actions + self.actions_good)

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
                for ratio, limit in zip(lineflow_simulate_ratio, thermal_limits):
                  if (limit < 400.00 and ratio > 0.90) or (limit >= 400.00 and ratio > 0.95):
                    has_danger = True

                if not done_simulate and score_simulate > max_score and not has_danger:
                  max_score = score_simulate
                  chosen_action = action
                  print('lineflow: ', lineflow_simulate_ratio)
                  print('current best action: {}, score: {:.4f}'.format(chosen_action, score_simulate))

            # chosen action
            action = chosen_action
          
          # count action
          count_action[action] = count_action.get(action, 0) + 1

          # take a step
          next_state, reward, done, info, flag_next_chronic = self.environment.step(
              self.action_space[action])

          score = self.get_score()

          if done:
            next_state = state
            score = -15
          else:
            next_state = np.array(next_state).reshape((-1, self.n_features))


          reward_tot = score / 15.0

          # record
          self.episode_score_history[episode] += score

          # Save transition to replay memory
          if done:
            # if done: store more
            for i in range(5):
              self.replay_memory.store(
                  [state, action, reward_tot, next_state, done])
          else:
            self.replay_memory.store(
                [state, action, reward_tot, next_state, done])

          # learn
          if step_tot > self.replay_memory_size and step_tot % 5 == 0:
            # Sample a minibatch from the replay memory
            tree_idx, batch_samples, IS_weights = self.replay_memory.sample(self.batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(
                np.array, zip(*batch_samples))
            states_batch = states_batch.reshape((-1, self.n_features))
            next_states_batch = next_states_batch.reshape(
                (-1, self.n_features))

            # Calculate q targets
            q_values_next = self.dqn_target.predict(sess, next_states_batch)
            q_values_next = np.array(q_values_next[0])
            targets_batch = reward_batch + \
                np.invert(done_batch).astype(np.float32) * \
                self.gamma * np.amax(q_values_next, axis=1)

            # Perform gradient descent update
            loss, abs_TD_errors = self.dqn_main.update(sess, states_batch, targets_batch,
                                                       action_batch.reshape((-1, 1)), IS_weights)
            # Update priority
            self.replay_memory.batch_update(tree_idx, abs_TD_errors)

          # verbose step summary
          if episode % self.verbose_per_episode == 0 and (step_tot + 1) % 1 == 0:
            print("episode: {}, step: {},  action: {}, loss: {:4f},  reward: {:4f}, score: {:.4f}, tot: {:.4f}\n".
                  format(episode + 1, step + 1, action, loss, reward, score, reward_tot))
            sys.stdout.flush()

          # update state
          state = next_state
          step_tot += 1

          if done or step_tot > total_train_step or flag_next_chronic:
            break

        # save model per episode
        self.saver.save(sess, self.model_dir + '/{}_model_176_step_{}_{}.ckpt'.format(isco, total_train_step, self.timestamp))
        print('Model Saved!')

        # verbose episode summary
        print("\nepisode: {}, mean_score: {:4f}, sum_score: {:4f}\n".
                  format(episode + 1, self.episode_score_history[episode] / (step + 1), self.episode_score_history[episode]))
        print("simulate used count: {}\naction count: {}\n".format(count_simulate[episode], sorted(count_action.items())))

        if step_tot > total_train_step:
          break

      time_end = time.time()
      print("\nFinished, Total time used: {}s".format(time_end - time_start))

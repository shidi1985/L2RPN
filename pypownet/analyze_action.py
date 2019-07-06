__author__ = 'tulan'
# Copyright (C) 2018 - 2019 GEIRI North America
# Authors: Tu Lan <tu.lan@geirina.net>

import numpy as np
import os
import glob
import seaborn
import matplotlib.pyplot as plt

# Directory
data_dir = os.path.abspath('data')
action_to_index_dir = os.path.join(data_dir, 'action_to_index.npy')
action_dir = os.path.join(data_dir, 'all_actions.npy')
reward_dir = os.path.join(data_dir, 'state_action_reward')
score_dir = os.path.join(data_dir, 'state_action_score')
tot_dir = os.path.join(data_dir, 'state_action_tot')
action_count_dir = os.path.join(score_dir, 'action_count_1000_tot.npy')


# generate action count based on tot reward
def generate_action_count_based_score():

  X_all = []
  y_all = []

  if not os.path.exists(action_count_dir):
    action_count = [0] * 3120
    for file in sorted(glob.glob(score_dir + '/*step_1000.npy')):
      print(file)
      try:
        state_action_reward = np.load(file, allow_pickle=True)
      except:
        print('cant load {}'.format(file))
        continue
      for k, v in state_action_reward.item().items():
        X_all.append(k)
        v = [x / 15.0  for x in v]
        y_all.append(v)
        chosen_action = np.argmax(v)
        action_count[chosen_action] += 1

    np.save(action_count_dir, action_count, allow_pickle=True)
    print('action count saved!')
  else:
    action_count = np.load(action_count_dir, allow_pickle=True)
    print('action count loaded')

  # prepare training data
  X_all = np.array(X_all, dtype=np.float32)
  y_all = np.array(y_all, dtype=np.float32)
  print(X_all.shape, y_all.shape)
  np.save(data_dir + '/X_all_251_{}_score.npy'.format(X_all.shape[0]), X_all, allow_pickle=True)
  np.save(data_dir + '/y_all_251_{}_score.npy'.format(X_all.shape[0]), y_all, allow_pickle=True)
  print('training data saved!')


# generate action_count or load
# generate imitation learning data
def generate_action_count_based_score_reward():

  X_all = []
  y_all = []

  if not os.path.exists(action_count_dir):
    action_count = [0] * 3120
    for file in sorted(glob.glob(reward_dir + '/*step_399.npy')):
      print(file)
      state_action_reward = np.load(file, allow_pickle=True)
      for k, v in state_action_reward.item().items():
        X_all.append(k)
        y_all.append(v)

    i = 0
    for file in sorted(glob.glob(score_dir + '/*step_399.npy')):
      print(file)
      state_action_score = np.load(file, allow_pickle=True)
      for k, v in state_action_score.item().items():
        # examine the same state or not
        if i < 3:
          print(sum(X_all[i]) == sum(k))
        y_all[i] = np.add(y_all[i], v)
        y_all[i] /= 10.0
        chosen_action = np.argmax(y_all[i])
        action_count[chosen_action] += 1
        i += 1


    np.save(action_count_dir, action_count, allow_pickle=True)
    print('action count saved!')
  else:
    action_count = np.load(action_count_dir, allow_pickle=True)
    print('action count loaded')

  # action taken > 0
  count_dict = {}
  for i, v in enumerate(action_count):
    if v > 0:
      count_dict[i] = v

  # prepare training data
  X_all = np.array(X_all, dtype=np.float32)
  y_all = np.array(y_all, dtype=np.float32)
  print(X_all.shape, y_all.shape)
  np.save(data_dir + '/X_all_{}.npy'.format(X_all.shape[0]), X_all, allow_pickle=True)
  np.save(data_dir + '/y_all_{}.npy'.format(X_all.shape[0]), y_all, allow_pickle=True)
  print('training data saved!')


def generate_action_count():

  X_all = []
  y_all = []

  if not os.path.exists(action_count_dir):
    action_count = [0] * 3120
    for file in sorted(glob.glob(reward_dir + '/*399.npy')):
      print(file)
      state_action_reward = np.load(file, allow_pickle=True)
      for k, v in state_action_reward.item().items():
        X_all.append(k)
        y_all.append(v)
        best_action = np.argmax(v)
        action_count[best_action] += 1
    np.save(action_count_dir, action_count, allow_pickle=True)
    print('action count saved!')
  else:
    action_count = np.load(action_count_dir, allow_pickle=True)
    print('action count loaded')

  # action taken > 0
  count_dict = {}
  for i, v in enumerate(action_count):
    if v > 0:
      count_dict[i] = v

  # prepare training data
  X_all = np.array(X_all, dtype=np.float32)
  y_all = np.array(y_all, dtype=np.float32)
  print(X_all.shape, y_all.shape)
  np.save(data_dir + '/X_all.npy', X_all, allow_pickle=True)
  np.save(data_dir + '/y_all.npy', y_all, allow_pickle=True)
  print('training data saved!')
  # plot barplot
  seaborn.barplot(list(count_dict.keys()), list(count_dict.values()))
  plt.title('Action Count')
  plt.xlabel('Action Index')
  plt.ylabel('Count')
  plt.show()


def analyze():
  action_count = np.load(action_count_dir, allow_pickle=True)
  print('action count loaded')

  # action taken > 0
  count_dict = {}
  for i, v in enumerate(action_count):
    if v > 0:
      count_dict[i] = v

  print('\n\n')
  for k, v in sorted(count_dict.items()):
    print('action: {}, count: {}'.format(k, v))

  # y = np.load(data_dir + '/y_all_3695_score.npy')
  # print(y[:20])

  # plot barplot
  seaborn.lineplot(list(count_dict.keys()), list(count_dict.values()))
  plt.title('Action Count')
  plt.xlabel('Action Index')
  plt.ylabel('Count')
  plt.show()


def analyze_action_count():
  action_space = np.load(action_dir, allow_pickle=True)
  n_actions = action_space.shape[0]
  y_all = np.load(data_dir + '/y_all_45125_score.npy', allow_pickle=True)
  print(y_all.shape)
  action_count = {}

  for x in y_all:
    best_action = np.argmax(x)
    action_count[best_action] = action_count.get(best_action, 0) + 1

  good_action ={}
  for k, v in action_count.items():
    if v >= 50 and 156 < k < 3100:
      good_action[k] = v

  print(good_action, len(good_action))

  actions_110 = np.array([action_space[k] for k in good_action])
  print(actions_110.shape)

  np.save(data_dir + '/actions_110.npy', actions_110, allow_pickle=True)
  print('action saved !')

def combine_action():
  actions_110 = np.load(data_dir + '/actions_110.npy', allow_pickle=True)
  actions_176 = np.load(data_dir + '/actions_176.npy', allow_pickle=True)
  actions_286 = np.concatenate([actions_176, actions_110])

  print(actions_286.shape)
  np.save(data_dir + '/actions_286.npy', actions_286, allow_pickle=True)
  print('action saved!')


if __name__ == "__main__":
  # load actions
  # action_space = np.load(action_dir, allow_pickle=True)
  # n_actions = action_space.shape[0]
  # action_to_index = np.load(action_to_index_dir, allow_pickle=True)

  generate_action_count_based_score()
  # generate_action_count_based_score_reward()
  # generate_action_count()
  analyze()
  # analyze_action_count()
  # combine_action()






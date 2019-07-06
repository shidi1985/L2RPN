__author__ = 'tulan'
# Copyright (C) 2018 - 2019 GEIRI North America
# Authors: Tu Lan <tu.lan@geirina.net>

import numpy as np
import os

# Directory
data_dir = os.path.abspath('data')

def generate_data():
  # load all data
  X_all = np.load(data_dir + '/X_all_251_40000_score.npy', allow_pickle=True)
  y_all = np.load(data_dir + '/y_all_251_40000_score.npy', allow_pickle=True)
  X_all_shape = X_all.shape
  y_all_shape = y_all.shape
  print(X_all_shape, y_all_shape)

  # shuffle index
  np.random.seed(22)
  random_index = list(range(X_all_shape[0]))
  np.random.shuffle(random_index)

  # seperate training and validation data
  train_end = int(X_all_shape[0] * 9.8 // 10)
  X_train = X_all[random_index[:train_end]]
  X_val = X_all[random_index[train_end:]]
  y_train = y_all[random_index[:train_end]]
  y_val = y_all[random_index[train_end:]]
  print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

  # verify 
  # print(X_all[random_index[5]] == X_train[5], y_all[random_index[5]] == y_train[5])

  # save data
  np.save(data_dir + '/X_train_{}.npy'.format(X_all_shape[0]), X_train, allow_pickle=True)
  np.save(data_dir + '/X_val_{}.npy'.format(X_all_shape[0]), X_val, allow_pickle=True)
  np.save(data_dir + '/y_train_{}.npy'.format(X_all_shape[0]), y_train, allow_pickle=True)
  np.save(data_dir + '/y_val_{}.npy'.format(X_all_shape[0]), y_val, allow_pickle=True)
  print('data saved!')

  X_train = np.load(data_dir + '/X_train_{}.npy'.format(X_all_shape[0]), allow_pickle=True)
  X_val = np.load(data_dir + '/X_val_{}.npy'.format(X_all_shape[0]), allow_pickle=True)
  y_train = np.load(data_dir + '/y_train_{}.npy'.format(X_all_shape[0]), allow_pickle=True)
  y_val = np.load(data_dir + '/y_val_{}.npy'.format(X_all_shape[0]), allow_pickle=True)
  print('data loaded')
  print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


if __name__ == "__main__":
  
  generate_data()

  # load all data
  # X_all = np.load(data_dir + '/X_all_15960.npy', allow_pickle=True)
  # y_all = np.load(data_dir + '/y_all_15960.npy', allow_pickle=True)
  # X_all_shape = X_all.shape
  # y_all_shape = y_all.shape
  # print(X_all_shape, y_all_shape)

  # X_train = np.load(data_dir + '/X_train_{}.npy'.format(X_all_shape[0]), allow_pickle=True)
  # X_val = np.load(data_dir + '/X_val_{}.npy'.format(X_all_shape[0]), allow_pickle=True)
  # y_train = np.load(data_dir + '/y_train_{}.npy'.format(X_all_shape[0]), allow_pickle=True)
  # y_val = np.load(data_dir + '/y_val_{}.npy'.format(X_all_shape[0]), allow_pickle=True)
  # print('data loaded')
  # print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

  # print(np.mean(y_train))



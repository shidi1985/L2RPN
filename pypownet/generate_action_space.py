__author__ = 'beizhang'
# Copyright (C) 2018 - 2019 GEIRI North America
# Authosr: Bei Zhang <bei.zhang@geirina.net>

import numpy as np
import os
import pypownet.environment
import pypownet.runner
import itertools
import copy


class Generate_Action_Space():
    def __init__ (self, parameters_folder=os.path.abspath('public_data'), game_level='datasets', chronic_looping_mode='natural', start_id=0,
                                          game_over_mode='soft'):
        self. environment = pypownet.environment.RunEnv(parameters_folder=os.path.abspath('public_data'),
                                          game_level='datasets',
                                          chronic_looping_mode='natural', start_id=0,
                                          game_over_mode='soft' )
        self.action_space = self.generate_action()
    
    def generate_action(self):
        sub_num = 14
        action_sapce_array = []
        action_space = self.environment.action_space
        observation_space = self.environment.observation_space
        game = self.environment.game
        
        for i in range(sub_num):
            action = action_space.get_do_nothing_action(as_class_Action=True)
            sub_id = i + 1
            if sub_id == 8: continue #ignore the change on bus 8
            expected_target_configuration = action_space.get_number_elements_of_substation(sub_id)    
            temp_space = list(itertools.product([0, 1], repeat = expected_target_configuration))
    
            n_space = len(temp_space)
            print (n_space)
    
            for i in range(int(n_space/2)):
                target_configuration = list(temp_space[i])
                if sum(target_configuration) == 0:
                    continue
                print ("substation: ", sub_id)
                print ("configuration: ", target_configuration)
                action_space.set_substation_switches_in_action(action, sub_id, target_configuration)
                action_sapce_array.append(action.as_array()[0:56])
        action = action_space.get_do_nothing_action(as_class_Action=True)
        action_sapce_array.append(action.as_array()[0:56])
        print (len(action_sapce_array))

        line_array = np.zeros(20)
        line_total_array = [line_array]
        for i in range(len(line_array)):
            if i == 13: continue #ignore the change on line F7-T8
            tmp = copy.deepcopy(line_array)
            tmp[i] = 1
            line_total_array.append(tmp)
        print (len(line_total_array))

        total_action_space = []
        for i in range(len(action_sapce_array)):
            tmp_sub = list(action_sapce_array[i])
            for j in range(len(line_total_array)):
                tmp_line = list(line_total_array[j])
                combine = copy.deepcopy(tmp_sub)
                combine.extend(tmp_line)
                combine = [int(i) for i in combine]

                total_action_space.append(combine)
        
        total_action_space = np.array(total_action_space)
        print (total_action_space.shape)
        return total_action_space


def generate_action():
    data_dir = os.path.abspath('public_data')
    action_dir = os.path.join(data_dir, 'all_actions.npy')
    action_to_index_dir = os.path.join(data_dir, 'action_to_index.npy')

    action_gen = Generate_Action_Space()
    action_to_index = {}
    for i, action in enumerate(action_gen.action_space):
        action_to_index[tuple(action)] = i
    np.save(action_dir, action_gen.action_space, allow_pickle=True)
    np.save(action_to_index_dir, action_to_index, allow_pickle=True)
    print('action saved!')
    loaded_action = np.load(action_dir, allow_pickle=True)
    loaded_action_to_index = np.load(action_to_index_dir, allow_pickle=True)
    print('action loaded! shape : {}'.format(loaded_action.shape))
    for k in loaded_action_to_index.item():
        print(loaded_action_to_index.item().get(k))

def generate_reduce_action():
    data_dir = os.path.abspath('data')
    action_dir = os.path.join(data_dir, 'all_actions.npy')

    # load action
    loaded_action = np.load(action_dir, allow_pickle=True)
    print('action loaded! shape : {}'.format(loaded_action.shape))

    action = np.concatenate((loaded_action[3100:], loaded_action[:156]))
    print(action.shape)

    save_action_dir = os.path.join(data_dir, 'actions_{}.npy'.format(action.shape[0]))

    np.save(save_action_dir, action, allow_pickle=True)
    x = np.load(save_action_dir, allow_pickle=True)
    print(x.shape)

if __name__ == '__main__':
    # generate_action()
    generate_reduce_action()

        
        

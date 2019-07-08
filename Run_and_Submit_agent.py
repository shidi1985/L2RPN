
import sys
import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sys import path
import time

# Useful paths for the submission and for data
model_dir = 'example_submission/'
ingestion_output = 'utils/logs/'

input_dir = 'public_data/'
output_dir = 'utils/output/'
results_dir = os.path.abspath('results')

# -
path.append(model_dir)
path.append(input_dir)
path.append(output_dir)

# from pypownet.runner import Runner
from codalab_tools.ingestion_program.runner import Runner #an override of pypownet.runner 
import pypownet.environment
import submission #existing exemple as submission.py

loads_p_file = '_N_loads_p.csv' #active power chronics for loads
prods_p_file = '_N_prods_p.csv'  #active power chronics for productions
datetimes_file = '_N_datetimes.csv' #timesstamps of the chronics
maintenance_file = 'maintenance.csv' #maintenance operation chronics. No maintenance considered in the first challenge
hazards_file = 'hazards.csv'   #harzard chronics that disconnect lines. No hazards considered in the first challenge
imaps_file = '_N_imaps.csv' #thermal limits of the lines
data_dir = 'public_data'  


def set_environement(game_level="datasets", start_id=0):
    """
        Load the first chronic (scenario) in the directory public_data/datasets 
    """
    return pypownet.environment.RunEnv(parameters_folder=os.path.abspath(input_dir),
                                              game_level=game_level,
                                              chronic_looping_mode='natural', start_id=start_id,
                                              game_over_mode="hard",)

def run_agent(environment, agent, number_iterations):
    # Instanciate a runner, that will save the run statistics within the log_path file, to be parsed and processed
    # by the scoring program
    log_path = os.path.abspath(os.path.join(ingestion_output, 'runner.log'))
    machinelog_filepath = os.path.abspath(os.path.join(ingestion_output,'machine_log.json'))# set to None 
    phase_runner = Runner(environment, submitted_controler, render=False, verbose=True, vverbose=False,
                                          log_filepath=log_path, machinelog_filepath=machinelog_filepath)
    phase_runner.ch.setLevel(logging.ERROR)
    # Run the planned experiment of this phase with the submitted model
    score, time, game_over_count = phase_runner.loop(iterations=number_iterations)
    print("cumulative reward : {}".format(score))
    return score, time, game_over_count


if not os.path.exists(ingestion_output):
    os.makedirs(ingestion_output)
log_path = os.path.abspath(os.path.join(ingestion_output, 'runner.log'))
open(log_path, 'w').close()



if __name__ == '__main__':

    NUMBER_ITERATIONS = 500 # The number of iterations can be changed
    n_test = 1
    label = 'test1'
    game_over_id = []
    game_over_count = 0
    score_history = []
    time_history = []


    for i in range(n_test):
        #or you can try your own submission
        # agent = submission.Submission
        agent = submission.Submission
        #agent = DoNothingAgent
        env = set_environement(start_id=i)

        submitted_controler = agent(env)
        score, time_used, game_over = run_agent(env, submitted_controler, NUMBER_ITERATIONS)

        # game over
        if game_over > 0:
            game_over_count += 1
            game_over_id += [i]
            score_history += [0]
        else:
            score_history += [score]

        time_history += [time_used]

        # print
        print('\n\n\nChronic: {}, Game Over: {}, Score: {:.3f}, Time: {:.3f}, Mean Score: {:.3f}, Mean Time: {:.3f}'.format(
            i, game_over, score, time_used, np.mean(score_history), np.mean(time_history)))

        # save res
        # np.save(results_dir + '/score_{}_chronic_{}.npy'.format(label, n_test), score_history)
        # np.save(results_dir + '/time_{}_chronic_{}.npy'.format(label, n_test), time_history)
        # print('Score and Time Saved!\n\n\n')
        sys.stdout.flush()

        time.sleep(1)

    # print
    print('\n\n\nFinished {} Chronics {} Steps, Mean Score: {:.3f}, Mean Time: {:.3f}, Game Over: {}\n\n\n'.format(
        n_test, NUMBER_ITERATIONS, np.mean(score_history), np.mean(time_history), game_over_id))
    print('Test Params: ', label)



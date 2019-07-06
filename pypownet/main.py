__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>, Tu Lan <tu.lan@geirina.net>

import argparse
from pypownet.environment import RunEnv
from pypownet.runner import Runner
import pypownet.agent

parser = argparse.ArgumentParser(description='CLI tool to run experiments using PyPowNet.')

parser.add_argument('-a', '--agent', metavar='AGENT_CLASS', default='DoNothing', type=str,
                    help='class to use for the agent (must be within the \'pypownet/agent.py\' file); '
                         'default class Agent')
parser.add_argument('-n', '--niter', type=int, metavar='NUMBER_ITERATIONS', default='1000',
                    help='number of iterations to simulate (default 1000)')
parser.add_argument('-p', '--parameters', metavar='PARAMETERS_FOLDER', default='./parameters/default14/', type=str,
                    help='parent folder containing the parameters of the simulator to be used (folder should contain '
                         'configuration.json and reference_grid.m)')
parser.add_argument('-lv', '--level', metavar='GAME_LEVEL', type=str, default='level0',
                    help='game level of the timestep entries to be played (default \'level0\')')
parser.add_argument('-s', '--start-id', metavar='CHRONIC_START_ID', type=int, default=0,
                    help='id of the first chronic to be played (default 0)')
parser.add_argument('-lm', '--loop-mode', metavar='CHRONIC_LOOP_MODE', type=str, default='natural',
                    help='the way the game will loop through chronics of the specified game level: "natural" will'
                         ' play chronic in alphabetical order, "random" will load random chronics ids and "fixed"'
                         ' will always play the same chronic folder (default "natural")')
parser.add_argument('-m', '--game-over-mode', metavar='GAME_OVER_MODE', type=str, default='soft',
                    help='game over mode to be played among "easy", "soft", "hard". With "easy", overflowed lines do '
                         'not break and game over do not end scenarios; '
                         'with "soft" overflowed lines are destroyed but game over do not end the scenarios; '
                         'with "hard" game over end the chronic upon game over signals and start the next ones if any.')
parser.add_argument('-r', '--render', action='store_true',
                    help='render the power network observation at each timestep (not available if --batch is not 1)')
parser.add_argument('-la', '--latency', type=float, default=None,
                    help='time to sleep after each frame plot of the renderer (in seconds); note: there are multiple'
                         ' frame plots per timestep (at least 2, varies)')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='display live info of the current experiment including reward, cumulative reward')
parser.add_argument('-vv', '--vverbose', action='store_true',
                    help='display live info + observations and actions played')

parser.add_argument('-f', '--function', metavar='RUN_FUNCTION', type=str, default='train',
                    help='choose main run function from imitation, train, and test')

def main():
  # Instantiate environment and agent
  args = parser.parse_args()
  env_class = RunEnv
  # agent_class = eval('pypownet.agent.{}'.format(args.agent))
  if args.game_over_mode.lower() not in ['easy', 'soft', 'hard']:
    raise ValueError('Unknown value {} for argument --game-over-mode; choices {}'.format(args.game_over_mode,
                                                                                         ['easy', 'soft', 'hard']))
  game_over_mode = 'hard' if args.game_over_mode.lower() == 'hard' else 'soft'
  without_overflow_cutoff = args.game_over_mode.lower() == 'easy'
  env = env_class(parameters_folder=args.parameters, game_level=args.level,
                  chronic_looping_mode=args.loop_mode, start_id=args.start_id,
                  game_over_mode=game_over_mode, renderer_latency=args.latency,
                  without_overflow_cutoff=without_overflow_cutoff)


  run_function = args.function.lower()

  if run_function == 'imitation':
    runner = Runner(env,
                args.render,
                args.verbose,
                args.vverbose,
                args.parameters,
                args.level,
                args.niter,
                n_features=538,
                n_episode=500,
                learning_rate=1e-3,
                gamma=0.99,
                replace_target_iter=128,
                replay_memory_size=512,
                PER_alpha=0.6,
                PER_beta=0.4,
                batch_size=1,
                epsilon_start=0.0,
                epsilon_end=0.0,
                verbose_per_episode=1,
                seed=22)

    runner.train_imitation(data_size=35380, test=False, model_name='new_model_176_imitation_35380_batch_1')

  elif run_function == 'train':
    runner = Runner(env,
                args.render,
                args.verbose,
                args.vverbose,
                args.parameters,
                args.level,
                args.niter,
                n_features=538,
                n_episode=10000,
                learning_rate=1e-4,
                gamma=0.99,
                replace_target_iter=128,
                replay_memory_size=256,
                PER_alpha=0.6,
                PER_beta=0.4,
                batch_size=32,
                epsilon_start=0.0,
                epsilon_end=0.0,
                verbose_per_episode=1,
                seed=22)

    runner.train(total_train_step=500000, model_name='model_176_imitation_35380_batch_1', isco='FC')

    # model_3120_imitation_45125
    # model_3120_step_1000000_06-07-17-56

  elif run_function == 'test':
    runner = Runner(env,
                args.render,
                args.verbose,
                args.vverbose,
                args.parameters,
                args.level,
                args.niter,
                n_features=538,
                n_episode=10,
                learning_rate=1e-4,
                gamma=0.99,
                replace_target_iter=128,
                replay_memory_size=512,
                PER_alpha=0.6,
                PER_beta=0.4,
                batch_size=128,
                epsilon_start=0.0,
                epsilon_end=0.0,
                verbose_per_episode=1,
                seed=22)

    runner.test(model_name='new_model_21_step_500000_06-13-21-03', episode_per_step=518)


if __name__ == "__main__":
  main()

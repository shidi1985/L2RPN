import numpy as np

import os
import sys
import yaml
import json
import itertools

from io import BytesIO
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import pypownet.environment
from pypownet.game import Game
from pypownet.environment import ActionSpace

def extract_json(file="machine_logs_0.json", action_space=None):
    with open(file, 'r') as json_file:
        datastore = json.load(json_file)
        labels = datastore["labels"]
        log = np.array(datastore["log"][0],dtype=object)

        build_plot(datastore,   action_space)

    game_overs = log[:, labels["game_over"]]
    if game_overs.sum() == 0:
        sc = sum([x[4] for x in log[:, labels["timestep_reward_aslist"]]])
        return sc
    else:
        return 0


def reward(log_filedir, filenames,action_space):
    cumulated_score = 0
    for file in filenames:
        r = extract_json(os.path.join(log_filedir, file),action_space)
        cumulated_score += r 

    return cumulated_score


def build_plot(data, action_space):
    reward_label = data["labels"]["cumulated_reward"]
    game_over_label = data["labels"]["game_over"] 
    rewards = np.array(data["log"][0],dtype=object)[:, reward_label]
    game_over = np.array(data["log"][0],dtype=object)[:, game_over_label]

    

    plt.figure(figsize=(10,7))
    # plt.pie(, labels=["Do_Nothing","Line Switch","Node splitting"], autopct='%1.1f%%', startangle=90)


    #plt.plot(rew)
    #for x in np.where(game_over)[0]:
    if sum(game_over):
        first_game_over = np.where(list(game_over))[0][0]
        #rew = [r if i < first_game_over else rewards[] for i, r in enumerate(rewards) ]
        plt.plot(rewards[:first_game_over])# list(range(first_game_over)))
        last = len(rewards)-first_game_over
        
        plt.axvline(x=first_game_over, color='r', linewidth=2)
        plt.plot(list(range(first_game_over, len(rewards),1)),[rewards[first_game_over]]*last,'--', color="g")

        plt.legend(["reward","game over"])
    else :
        plt.plot(rewards)
        plt.legend(["reward"])
    score = rewards[-1] if sum(game_over)==0 else 0
    n_game_over = sum(game_over)
    plt.title("cummulated reward over time final score :{:.0f}, had game_over : {}".format(score, (0 != n_game_over)))
    plt.xlabel('timestep')
    plt.ylabel('cummulated Reward')
    plt.legend(["reward", "game over"])

    if action_space is not None:
        action_label = data["labels"]["action"]
        actions = np.array(np.array(data["log"][0],dtype=object)[:, action_label])

        action_counter, count_three_types = action_count(action_space, actions)



        plt.figure(figsize=(5,5))
        plt.pie(count_three_types, labels=["Do Nothing", "Line action", "Node splitting"], startangle=90,autopct='%1.1f%%')
        plt.title("Distribution of type action types")


        plt.figure(figsize=(10,7))
        x = list(action_counter.keys())[0:21]
        h = [action_counter[el] for el in x]
        plt.bar(x, h, 1)
        plt.title("Distribution of line actions")
        plt.xticks(rotation=45)


        plt.figure(figsize=(10,7))
        x = ["Do Nothing"] + list(action_counter.keys())[21:]
        h = [action_counter[el] for el in x]
        plt.bar(x, h, 1)
        plt.title("Distribution of substations actions")
        plt.xticks(rotation=45)
 


def save_figures():
    fig_list = list()
    for i in plt.get_fignums():
        fig = plt.figure(i)
        figfile = BytesIO()
        # fig.set_size_inches(12)
        plt.savefig(figfile, format='png')  # ,dpi=100)
        figfile.seek(0)
        fig_list.append(base64.b64encode(figfile.getvalue()).decode('ascii'))
        plt.close(fig)
    return fig_list


def html_text(fig_list):
    html = """<html><head></head><body>\n"""
    for i,figure in enumerate(fig_list):
        if i %4==0 :
            html+= "<hr> scenario {}<br>".format(i//4)
        html += '<img src="data:image/png;base64,{0}"><br>'.format(figure)

    html += """</body></html>"""

    return html





def get_action_space(in_dir):
    game = Game(parameters_folder=in_dir, game_level="ref_shape",
                chronic_looping_mode="natural", chronic_starting_id=0,
                game_over_mode="soft", renderer_frame_latency=None, without_overflow_cutoff=False)

    action_space = ActionSpace(*game.get_number_elements(),
                               substations_ids=game.get_substations_ids(),
                               prods_subs_ids=game.get_substations_ids_prods(),
                               loads_subs_ids=game.get_substations_ids_loads(),
                               lines_or_subs_id=game.get_substations_ids_lines_or(),
                               lines_ex_subs_id=game.get_substations_ids_lines_ex())

    return action_space




def action_count(action_space, actions):
    possible_actions = list_possible_actions(action_space)
    action_counter = dict.fromkeys(list(possible_actions.keys()), 0)

    for action in actions:
        for action_type in possible_actions.keys():
            action_counter[action_type] += (tuple(action) in possible_actions[action_type])

    count_three_types = [
        action_counter["Do Nothing"],
        sum([action_counter["line : {:.0f}".format(i) ] for i in range(action_space.lines_status_subaction_length)]),
        sum([action_counter["sub : {:.0f}".format(s) ] for s in action_space.substations_ids]),
    ]

    return action_counter, count_three_types


def list_possible_actions(action_space):
    """ lists all unary actions"""

    number_lines = action_space.lines_status_subaction_length
    possible_actions = dict.fromkeys(["Do Nothing"] +
                                     ["line : {:.0f}".format(i)  for i in range(number_lines)] +
                                     ["sub : {:.0f}".format(s) for s in action_space.substations_ids],
                                     set())
    possible_actions["Do Nothing"] = {tuple(action_space.get_do_nothing_action())}
    # actions.append(n(ap.get_do_nothing_action(),dtype=torch.float))
    for l in range(number_lines):
        action = action_space.get_do_nothing_action(as_class_Action =True)
        action_space.set_lines_status_switch_from_id(action=action, line_id=l, new_switch_value=1)
        possible_actions["line : {:.0f}".format(l) ] = {tuple(action.as_array())}
    # substation_actions = dict()
    for substation_id in action_space.substations_ids:
        substation_actions = set()
        substation_n_elements = action_space.get_number_elements_of_substation(substation_id)
        for configuration in list(itertools.product([0, 1], repeat=substation_n_elements))[1:]:
            action = action_space.get_do_nothing_action(as_class_Action=True)
            action_space.set_substation_switches_in_action(action=action, substation_id=substation_id,
                                                           new_values=configuration)
            substation_actions.add(tuple(action.as_array()))
        possible_actions["sub : {:.0f}".format(substation_id)] = substation_actions
    return possible_actions





def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    print(input_dir)
    print(output_dir)


    submit_dir = os.path.join(input_dir, 'res')
    action_space = get_action_space(os.path.join(input_dir,'ref'))
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.abspath(os.path.join(submit_dir, 'log_files.json')), 'r') as files:
            files = json.load(files)

            score = reward(submit_dir, files, action_space)

            try:
                metadata = yaml.load(open(os.path.join(submit_dir, 'metadata'), 'r'), Loader=yaml.FullLoader)
                duration = metadata['elapsedTime']
            except:
                duration = 0

            output_filename = os.path.join(output_dir, 'scores.txt')
            with open(output_filename, 'w') as f:
                f.write("score: {}\n".format(score))
                f.write("Duration: %0.6f\n" % duration)
                f.close()

            figure_list = save_figures()
            output_filename = os.path.join(output_dir, 'scores.html')
            with open(output_filename, 'w') as f:

                # f.write("score: {}\n".format(score))
                # f.write("plots incoming")
                f.write(html_text(figure_list))
                f.close()
            # print("step : {}, cumulative rewards : {}".format(step,cumulative_reward ))



if __name__ == "__main__":
    main()

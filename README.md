# Learning to Run a Power Network
https://competitions.codalab.org/competitions/22845#learn_the_details-overview

## Pypownet Installation
### Requirements:
*   Python >= 3.6
*   Virtual Environment Recommended

```
cd L2RPN/
python setup.py install
```
Pypownet Introduction: https://github.com/MarvinLer/pypownet

## Basic Usage
### Runing Test
```
python Run_and_Submit_agent.py
```
You can modify the testing number of chronics and timesteps in the 'Run_and_Submit_agent.py' file.

### Train Your Model
```
python -m pypownet.main -f train
```
To see all the options:
```
python -m pypownet.main --help
```

## Key Files and Features
- data
    - Saved numpy files of action_space and generated train/val data
    - Trained model
- example_submission
    - Sample submission to the L2RPN competition
- parameters
    - reward_signal, configuration, and training chronics of different power grids
- public_data
    - Extra data for IEEE-14 bus
- pypownet
    - agent.py: Defines the Dueling DQN agent
    - analyze_action.py: Analyze the simulation results
    - generate_action_space.py: Generate action space
    - main.py: Main run file, including imitation, training, and test
    - prepare_data.py: prepare data for imitation learning
    - runner.py: key file controling the training process
- Run_and_Submit_agent.py
    - Test the trained model

# License information

Copyright 2017-2019 GEIRINA, RTE, and INRIA (France)
    
    GEIRINA: https://www.geirina.net/
    RTE: http://www.rte-france.com
    INRIA: https://www.inria.fr/

This Source Code is subject to the terms of the GNU Lesser General Public License v3.0. If a copy of the LGPL-v3 was not distributed with this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.fr.html.

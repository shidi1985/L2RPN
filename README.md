# Learning to Run a Power Network
https://competitions.codalab.org/competitions/22845#learn_the_details-overview

## Pypownet Installation
### Requirements:
*   Python >= 3.6
*   Virtual Environment Recomended

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
python -m pypowenet --help
```

# License information

Copyright 2017-2019 GEIRINA, RTE, and INRIA (France)
    
    GEIRINA: https://www.geirina.net/
    RTE: http://www.rte-france.com
    INRIA: https://www.inria.fr/

This Source Code is subject to the terms of the GNU Lesser General Public License v3.0. If a copy of the LGPL-v3 was not distributed with this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.fr.html.

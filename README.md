
<p align="center">
  <img width="50%" src="./.docs/source/_static/deepchain.png">
</p>

![PyPI](https://img.shields.io/pypi/v/deepchain-apps)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![Documentation Status](https://readthedocs.org/projects/deepchain-apps/badge/?version=latest)](https://deepchain-apps.readthedocs.io/en/latest/?badge=latest)

<details><summary>Table of contents</summary>

- [Description](#description)
- [How it works](#howitworks)
- [Getting started with App](#usage)
- [CLI](#usage)
  - login
  - create
  - deploy
  - apps
- [Roadmap](#roadmap)
- [Citations](#citations)
- [License](#license)
</details>


# Description
DeepChain apps is a collaborative framework that allows the user to create scorers to evaluate protein sequences. These scorers can be either classifier or predictor.

This Github is hosting a template for creating a personal application to deploy on deepchain.bio. The main [deepchain-apps](https://pypi.org/project/deepchain-apps/) package can be found on pypi.
To leverage the apps capability, take a look at the [bio-transformers](https://pypi.org/project/bio-transformers/) and [bio-datasets](https://pypi.org/project/bio-datasets) package.

📕 Please find the documentation [here](https://deepchain-apps.readthedocs.io/en/latest/index.html).

## Installation
It is recommended to work with conda environments in order to manage the specific dependencies of the package.

```bash
  conda create --name deepchain-env python=3.7 -y 
  conda activate deepchain-env
  pip install deepchain-apps
```

# How it works
If you want to create and deploy an app on deepchain hub, you could use the command provided in the [deepchain-apps](https://pypi.org/project/deepchain-apps/) package.
Below are the main commands that should be used in a terminal:

## Basic CLI

```
deepchain login
deepchain create myapplication
```

The last command will download the Github files inside the **myapplication** folder.

You can modify the app.py file, as explained in the [Deepchain-apps templates](#deepchain-apps-templates)

To deploy the app on deepchain.bio, use:

```
deepchain deploy myapplication
```

To know how to generate a token with deepchain, please follow this [link](https://deepchain-apps.readthedocs.io/en/latest/documentation/deepchain.html)

# App structure
When creating an app, you will download the current Github folder with the following structure.

```bash
 .
├── README.md # explains how to create an app
├── __init__.py # __init__ file to create python module
├── checkpoint
│   ├── __init__.py
│   └── Optionnal : model.pt # optional: model to be used in app must be placed there
├── examples
│   ├── app_with_checkpoint.py # example: app example with checkpoint
│   └── torch_classifier.py # example: show how to train a neural network with pre-trained embeddings
└── src
    ├── DESC.md # Desciption file of the application, feel free to put a maximum of information.
    ├── __init__.py
    ├── app.py # main application script. Main class must be named App.
    └── Optional : model.py # file to register the models you use in app.py.
    └── tags.json # file to register the tags on the hub.
    
```

The main class must be named ```App``` in ```app.py```

### Tags
For your app to be visible and well documented, tags should be filled to precise at least the *tasks* section.
It will be really useful to retrieve it from deepchain hub.

  - tasks
  - librairies
  - embeddings
  - datasets
  - device

If you want your app to benefit from deepchain' GPU, set the device to "gpu" in tags. It will run on "cpu" by default.

# Deepchain-apps templates

You can also create an application based on an app already available on the public [deepchain hub](https://app.deepchain.bio/hub/apps):

## Apps from deepchain hub

First, you can list all the available app in the hub like following:

```
>> deepchain apps --public

----------------------------------------------------------------
APP                                        USERNAME             
----------------------------------------------------------------
OntologyPredict                    username1@instadeep.com    
DiseaseRiskApp                     username2@instadeep.com     
```

You can simply download the app locally with the cli:

```
deepchain download username1@instadeep.com/OntologyPredict OntologyPredict
```

The app will be downloaded in the OntologyPredict folder.

## Templates
Some templates are provided to create and deploy an app.

You can implement whatever function you want inside ```compute_scores()``` function. 

It just has to respect the return format: 

One dictionary for each protein that is scored. Each key of the dictionary are declared in ```score_names()``` function.

```python
[
  {
    'score_names_1':score11
    'score_names_2':score21
  },
   {
    'score_names_1':score12
    'score_names_2':score22
  }
]
```

## Scorer based on a neural network
An example of training with an embedding is provided in the example/torch_classifier.py script.

Be careful, you must use the same embedding for the training and the ```compute_scores()``` method.

### Where to put the model?
When training a model with pytorch, you must save the weights with the ```state_dict()``` method, rebuilt the model architecture in the Scorer or in a ```model.py``` file and load the weights like in the example below.

```python
from typing import Dict, List, Optional

import torch
from biotransformers import BioTransformers
from deepchain.components import DeepChainApp

# TODO : from model import myModel
from deepchain.models import MLP
from torch import load

Score = Dict[str, float]
ScoreList = List[Score]


class App(DeepChainApp):
    """DeepChain App template:

    - Implement score_names() and compute_score() methods.
    - Choose a a transformer available on BioTranfformers
    - Choose a personal keras/tensorflow model
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.num_gpus = 0 if device == "cpu" else 1
        self.transformer = BioTransformers(backend="protbert", num_gpus=self.num_gpus)
        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = "model.pt"
        # build your model
        self.model = MLP(input_shape=1024, n_class=2)

        # load_model for tensorflow/keras model-load for pytorch model
        if self._checkpoint_filename is not None:
            state_dict = load(self.get_checkpoint_path(__file__))
            self.model.load_state_dict(state_dict)
            self.model.eval()

    @staticmethod
    def score_names() -> List[str]:
        """App Score Names. Must be specified.

        Example:
         return ["max_probability", "min_probability"]
        """
        return ["probability"]

    def compute_scores(self, sequences: List[str]) -> ScoreList:
        """Return a list of all proteins score"""

        x_embedding = self.transformer.compute_embeddings(sequences)["cls"]
        probabilities = self.model(torch.tensor(x_embedding).float())
        probabilities = probabilities.detach().cpu().numpy()

        prob_list = [{self.score_names()[0]: prob[0]} for prob in probabilities]

        return prob_list
```

# Getting started with deepchain-apps cli

##  CLI
The CLI provides 5 main commands:

- **login** : you need to supply the token provide on the platform (PAT: personal access token).

  ```
  deepchain login
  ```

- **create** : create a folder with a template app file

  ```
  deepchain create my_application
  ```

- **deploy** : the code and checkpoint are deployed on the platform, you can select your app in the interface on the platform.
  - with checkpoint upload

    ```
    deepchain deploy my_application --checkpoint
    ```

  - Only the code

    ```
    deepchain deploy my_application
    ```

- **apps** :
  - Get info on all local/upload apps

    ```
    deepchain apps --infos
    ```

  - Remove all local apps (files & config):

    ```
    deepchain apps --reset
    ```

  - Remove a specific application (files & config):

    ```
    deepchain apps --delete my_application
    ```

  - List all public apps:

    ```
    deepchain apps --public
    ```

- **download** :
  - Download locally an app deployed on deepchain hub

    ```
      deepchain download user.name@mail.com/AppName AppName
    ```


## License
Apache License Version 2.0


<p align="center">
  <img width="50%" src="./.docs/source/_static/logo-instadeep-longeur.png">
</p>

# Description
DeepChain apps is a collaborative framework that allows the user to create scorers to evaluate protein sequences. These scorers can be either Classifiers or Predictors. 

This github is hosting a template for creating a personnal application to deploy on deepchain.bio. The main deepchain-apps package can be found on pypi.

## Installation
It is recommmanded to work with conda environnements in order to manage the specific dependencies of the package.
```bash
  conda create --name deepchain-env python=3.7 -y 
  conda activate deepchain-env
  pip install deepchain-apps
```

# How it works
Some command are provided in order to create and deploy an application. Below are the main command that should be used in a terminal:

```
deepchain login
deepchain create myapplication
```
The last command will download the github files inside the myapplication folder.

You can modify the app.py file, as explain in the [Deechain-apps templates](#deepchain-apps-templates)

To deploy the app on deepchain.bio, use:

```
deepchain deploy myapplication
```


## App structure

This template provide an example of application that you can submit.
The final app must have the following architecture:

- my_application
  - src/
    - app.py
    - Optionnal : requirements.txt (for extra packages)
  - checkpoint/
    - Optionnal : model.[h5/pt]

The main app class must be named ’App’

# Deepchain-apps templates

Some templates are provided in order to create and deploy an app.
## Examples

You can  implement whatever function you want inside ```compute_scores()``` function. 

It just have to respect to return format: 
One dictionnary for each proteins that are scored. Each keys of the dictionnary are declared in ```score_names()``` function.

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

### Neural Network scorer

```python
from pathlib import Path
from typing import Dict, List, Optional

import torch
from deepchain.components import DeepChainApp, Transformers
from torch import load

Score = Dict[str, float]
ScoreList = List[Score]


class App(DeepChainApp):
    """
    DeepChain App template:
        Implement score_names() and compute_score() methods.
        Choose a a transformer available on DeepChain
        Choose a personal keras/tensorflow model
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.transformer = Transformers(device=device, model_type="esm1_t6_43M_UR50S")

        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = "model.pt"

        # Use load_model for tensorflow/keras model
        # Use load for pytorch model
        if self._checkpoint_filename is not None:
            self.model = load(self.get_checkpoint_path(__file__))

    @staticmethod
    def score_names() -> List[str]:
        """
        App Score Names. Must be specified.

        Example:
         return ["max_probability", "min_probability"]
        """
        return ["probability"]

    def compute_scores(self, sequences: List[str]) -> ScoreList:
        """
        Return a list of all proteins score
        Score must be a list of dict:
                - element of list is protein score
                - key of dict are score_names, value is the scorexs

        Example:
            Calculate embeddings with the pre-trained Tranformer module
            -- Use same embedding as the one for training the model!
            -- Get available embedding with :
                >> transformer.list_esm_backend()
                >> embeddings = self.transformer.predict_embedding(sequences)
        """
        if not isinstance(sequences, list):
            sequences = [sequences]

        x_embedding = self.transformer.predict_embedding(sequences)
        probabilities = self.model(torch.tensor(x_embedding).float())
        probabilities = probabilities.detach().cpu().numpy()

        prob_list = [{self.score_names()[0]: prob[0]} for prob in probabilities]

        return prob_list
```

# Getting started with deepchain-apps CLI

##  Command
The CLI provides 4 main commands:

- **login** : you need to supply the token provide on the platform (PAT: personnal access token).

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

The application will be deploy in DeepChain platform.

## Embedding

Some embeddings are provided in the `Transformers` module

```python
from deepchain.components import Transformers
```

The model are furnished, but not mandatory, if you want to make an embedding of your protein sequence.
Only the ESM (evolutionary scale modeling) model is provided, with different architecture.
Here for some full details of the architecture (https://github.com/facebookresearch/esm)

- 'esm1_t6_43M_UR50S'
- 'esm1_t12_85M_UR50S'
- 'esm_msa1_t12_100M_UR50S'
- 'esm1b_t33_650M_UR50S'
- 'esm1_t34_670M_UR100'
- 'esm1_t34_670M_UR50D'
- 'esm1_t34_670M_UR50S'

!! The embedding will run on a GPU on the platform. But for a testing phase on your personal computer (CPU), you should choose the smaller architecture.

## License
Apache License Version 2.0
# Template for creating a personnal application on deepchain.bio plateform

## Getting started with App

This template provide an example of application that you can submit.
The final app must have the following architecture:

- my_application
  - src/
    - app.py
    - Optionnal : requirements.txt (for extra packages)
  - checkpoint/
    - Optionnal : model.[h5/pt]

The main app class must be named ’App’

## CLI

The CLI provides 4 main commands:

- **login** : you need to supply the token provide on the plateform (PAT: personnal access token).

  ```
  deepchain login
  ```

- **create** : create a folder with a template app file

  ```
  deepchain create my_application
  ```

- **deploy** : the code and checkpoint are deployed on the plateform, you can select your app in the interface on the plateform.
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

The application will be deploy in DeepChain plateform.

## Embedding

Some embeddings are provided in the `Transformers` module

```
from deepchainapps.components import Transformers
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

!! The embedding will run on a GPU on the plateform. But for a testing phase on your personal computer (CPU), you should choose the smaller architecture.

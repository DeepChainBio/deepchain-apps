=========
Basic app
=========

As explained `here <https://deepchain-apps.readthedocs.io/en/latest/documentation/deepchainapps.html>`_, an app has the stucture of a python package.
Every files should be placed in the ``src`` or ``checkpoint`` folder. 

The app should at least contains an ``app.py`` file with an ``App`` class in the ``src`` folder. The provided template is when creating an app 
is a starting point. Note that the ``compute_scores`` function receive a list of sequences in input.


Template
--------

The app below describes the general framework to build an app.

.. code-block:: python

    from typing import Dict, List, Optional
    from collections import Counter
    from biotransformers import BioTransformers
    from deepchain.components import DeepChainApp
    from torch import load

    Score = Dict[str, float]
    ScoreList = List[Score]


    class App(DeepChainApp):
        """DeepChain App template:
        * Implement score_names() and compute_score() methods.
        * Choose a transformer available on bio-transformers (or others pacakge)
        * Choose a personal keras/tensorflow model (or not)
        * Build model and load the weights.
        * compute whatever score of interest based on protein sequence
        """

        def __init__(self, device: str = "cpu"):
            self._device = device

            # TODO: fill _checkpoint_filename if needed
            # Make sure to put your checkpoint file in your_app/checkpoint folder
            self._checkpoint_filename: Optional[str] = None

            # TODO:  Use proper loading function
            # load_model for tensorflow/keras model - load for pytorch model
            # torch model must be built before loading state_dict
            if self._checkpoint_filename is not None:
                #state_dict = load(self.get_checkpoint_path(__file__))
                # self.model.load_state_dict(state_dict)
                pass

        @staticmethod
        def score_names() -> List[str]:
            """App Score Names. Must be specified
            Returns:
                A list of score names
            Example:
                return ["max_probability", "min_probability"]
            """
            # TODO : Put your own score_names here
            # For this template, we just count the number of A in the sequence.
            return ["A_count"]

        def compute_scores(self, sequences: List[str]) -> ScoreList:
            """Compute a score based on a user defines function.
            This function compute a score for each sequences receive in the input list.
            Caution :  to load extra file, put it in src/ folder and use
                    self.get_filepath(__file__, "extra_file.ext")
            Returns:
                ScoreList object
                Score must be a list of dict:
                        * element of list is protein score
                        * key of dict are score_names
            """
            # TODO : Fill with you own score function
            count_A = [Counter(seq).get("A",0) for seq in sequences]
            score_list = [{self.score_names()[0]: count} for count in count_A]

            return score_list

ScoreList
---------

All the application must have a ``compute_scores()`` method. The function can return a list of score that will be selectable in deepchain
to use in the optimizer. The scores' names that will appear in ``deepchain`` optimizer have to be put in ``score_names()`` function.

The return of the ``compute_scores()`` must be a list of dict, where each dict correspond to a protein score, and each key of the dict to 
a score names.

.. code-block:: python

    [
    {
        'score_names_1':score1_seq1
        'score_names_2':score2_seq1
    },
    {
        'score_names_1':score1_seq2
        'score_names_2':score2_seq2
    }
    ,...
    {
        'score_names_1':score1_seqn
        'score_names_2':score2_seqn
    }
    ]

App with model
--------------

 You have the ability to build an app with a model checkpoint in pytorch or tensorflow.

.. WARNING::  You must build your model inside the ``app.py`` file or put a ``model.py`` inside the ``src`` folder and import it. You have to load the ``state_dict`` in the model with pytorch.
.. Hint:: The embeddings in the example below are computed with ``bio-transformers`` and the ``MLP`` is imported from ``deepchain``. There is no restriction about the kind of model to use.

.. code-block:: python

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
            self.num_gpus = 1 if device == "cpu" else 1
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

Training a model
----------------

.. Important::  When working with pytorch, you must save your model with ``state_dict`` as explained `here <https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended>`_, and reload it inside the app.

You can build the model of your choice, from embeddings or not, and load it in your app.

.. code-block:: python

    """
    A module that provides a classifier template to train a model on embeddings.
    With using the pathogen vs human dataset as an example. The embedding of 100k proteins come from the protBert model.
    The model is built with pytorch_ligthning, a wrapper on top of 
    pytorch (similar to keras with tensorflow)
    Feel feel to build your own model if you want to build a more complex one
    """

    import numpy as np
    from biodatasets import list_datasets, load_dataset
    from deepchain.models import MLP
    from deepchain.models.utils import confusion_matrix_plot, model_evaluation_accuracy
    from sklearn.model_selection import train_test_split

    # Load embedding and target dataset
    pathogen = load_dataset("pathogen")
    _, y = pathogen.to_npy_arrays(input_names=["sequence"], target_names=["class"])
    embeddings = pathogen.get_embeddings("sequence", "protbert", "cls")

    x_train, x_test, y_train, y_test = train_test_split(embeddings, y[0], test_size=0.3)

    # Build a multi-layer-perceptron on top of embedding

    # The fit method can handle all the arguments available in the
    # 'trainer' class of pytorch lightening :
    #               https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
    # Example arguments:
    # * specifies all GPUs regardless of its availability :
    #               Trainer(gpus=-1, auto_select_gpus=False, max_epochs=20)

    # Input variables for MLP
    n_class = len(np.unique(y_train))
    input_shape = x_train.shape[1]

    mlp = MLP(input_shape=input_shape, n_class=n_class)
    mlp.fit(x_train, y_train, epochs=5)
    mlp.save("model.pt") # built-in method to save state_dict

    # Model evaluation
    y_pred = mlp(x_test).squeeze().detach().numpy()
    model_evaluation_accuracy(y_test, y_pred)

    # Plot confusion matrix
    confusion_matrix_plot(y_test, (y_pred > 0.5).astype(int), ["0", "1"])
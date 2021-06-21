==========
Apps basic
==========

Template
--------

The app below describes the general framework to build an app.

.. code-block::python

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

        def __init__(self, device: str = "cuda:0"):
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

.. code-block::python
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

Score with model
----------------
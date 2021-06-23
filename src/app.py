"""Template file to develop personal app
WARNINGS: if you run the app locally and don't have a GPU you should choose device='cpu'
"""

from typing import Dict, List, Optional

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
        self.num_gpus = 0 if device == "cpu" else 1
        self.transformer = BioTransformers(backend="protbert", num_gpus=self.num_gpus)

        # TODO: fill _checkpoint_filename if needed
        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = None

        # TODO:  Use proper loading function
        # load_model for tensorflow/keras model - load for pytorch model
        # torch model must be built before loading state_dict
        if self._checkpoint_filename is not None:
            state_dict = load(self.get_checkpoint_path(__file__))
            # self.model.load_state_dict(state_dict)

    @staticmethod
    def score_names() -> List[str]:
        """App Score Names. Must be specified

        Returns:
            A list of score names

        Example:
            return ["max_probability", "min_probability"]
        """
        # TODO : Put your own score_names here
        return ["loglikelihood"]

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
        loglikelihoods = self.transformer.compute_loglikelihood(sequences)
        log_list = [{self.score_names()[0]: log} for log in loglikelihoods]

        return log_list


if __name__ == "__main__":

    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]
    app = App("cpu")
    scores = app.compute_scores(sequences)
    print(scores)

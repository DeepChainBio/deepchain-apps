"""Template file to develop personal app
WARNINGS: if you run the app locally and don't have a GPU
          you should choose device='cpu'
"""

from typing import Dict, List, Optional

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
        self.transformer = Transformers(device=device)

        # TODO: fill _checkpoint_filename if needed
        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = None

        # TODO:  Use proper loading function
        #        load_model for tensorflow/keras model
        #        load for pytorch model
        if self._checkpoint_filename is not None:
            self.model = load(self.get_checkpoint_path(__file__))

    @staticmethod
    def score_names() -> List[str]:
        """
        App Score Names. Must be specified
        Example:
         return ["max_probability", "min_probability"]
        """
        # TODO : Put your own score_names here
        return ["loglikelihood"]

    def compute_scores(self, sequences: List[str]) -> ScoreList:
        """
        Return a list of all proteins score
        Score must be a list of dict:
                - element of list is protein score
                - key of dict are score_names

        Example:
            Calculate embeddings with the pre-trained transformer
            >> embeddings = self.transformer.predict_embedding(sequences)
        """

        # TODO : Fill with you own score function

        if not isinstance(sequences, list):
            sequences = [sequences]

        loglikelihoods = self.transformer.predict_loglikelihood(sequences)
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

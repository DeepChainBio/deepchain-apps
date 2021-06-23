"""Template file to develop personal app
WARNINGS: if you run the app locally and don't have a GPU
          you should choose device='cpu'
"""

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

        # TODO:  Use proper loading function
        # load_model for tensorflow/keras model - load for pytorch model
        # torch model must be built before loading state_dict
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


if __name__ == "__main__":

    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]
    app = App("cpu")
    scores = app.compute_scores(sequences)
    print(scores)

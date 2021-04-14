"""Template file to develop personal app
WARNINGS: if you run the app locally and don't have a GPU
          you should choose device='cpu'
"""


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

        # TODO  FILL _checkpoint_filename if needed
        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = "model.pt"

        # TODO  Use proper loading function
        # load_model for tensorflow/keras model - load for pytorch model
        if self._checkpoint_filename is not None:
            self.model = load(self.get_checkpoint_path(__file__))

    @staticmethod
    def score_names() -> List[str]:
        """
        App Score Names. Must be specified.

        Example:
         return ["max_probability", "min_probability"]
        """
        # TODO : Put your own score_names here
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
        # TODO : Fill with you own score function

        if not isinstance(sequences, list):
            sequences = [sequences]

        x_embedding = self.transformer.predict_embedding(sequences)
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

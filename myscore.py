"""Template file to develop personal scorer
WARNINGS: if you run the scorer locally and don't have a GPU
          you should choose device='cpu'
"""


from typing import Dict, List

import numpy as np
from deepchainapps.components import TransformersApp
from tensorflow.keras.models import load_model
from torch import load

from base_scorer import UserScorer


class Myscorer(UserScorer):
    """
    Scorer template:
    criteria and compute_score methods are mandatory
    Feel free to choose an embedding that will run on the plateform
    and a personal keras/tensorflow model
    """

    def __init__(self, checkpoint_path: str = None, device: str = "cuda:0"):
        self._checkpoint_path = checkpoint_path
        self._device = device
        self.app = TransformersApp(device=device)

        if checkpoint_path is not None:
            self.model = load_model(checkpoint_path)

    @property
    def criteria(self) -> List[str]:
        """
        Criteria names.

        Example:

         return ["max_probability", "min_probability"]

        """
        return ["max_probability", "min_probability"]

    def compute_scores(self, sequences: List[str]) -> List[Dict[str, float]]:
        """
        Return a list of all proteins score
        Score must be a list of dict:
                - element of list is protein score
                - key of dict are criterias


        >> embedding example:

        emb_vector = self.emb_model.predict_embedding(sequences)

        >> probabilities prediction (previously train model)

        probabilities = self.model.predict(emb_vector)

        >> score formating:

        scores = [
            {
                self.criteria[0]: float(np.max(probabilities)),
                self.criteria[1]: float(np.min(probabilities)),
            }
            for seq, prob in probabilities
        ]

        return scores

        """
        if not isinstance(sequences, list):
            sequences = [sequences]

        probabilities = self.model.predict(sequences)

        scores = [
            {
                self.criteria[0]: 0,
                self.criteria[1]: 0,
            }
            for _ in probabilities
        ]

        return scores

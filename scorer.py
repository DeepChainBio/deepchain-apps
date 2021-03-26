"""Template file to develop personal scorer
WARNINGS: if you run the scorer locally and don't have a GPU
          you should choose device='cpu'
"""


from typing import Dict, List

from deepchainapps.components import TransformersApp, UserScorer
from tensorflow.keras.models import load_model

Score = Dict[str, float]
ScoreList = List[Score]


class Scorer(UserScorer):
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

    @staticmethod
    def score_names() -> List[str]:
        """
        Criteria names.

        Example:

         return ["max_probability", "min_probability"]

        """
        return ["loglikelihood"]

    def compute_scores(self, sequences: List[str]) -> ScoreList:
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
                self.criteria()[0]: float(np.max(prob)),
                self.criteria()[1]: float(np.min(prob)),
            }
            for prob in probabilities
        ]

        return scores

        """
        if not isinstance(sequences, list):
            sequences = [sequences]

        loglikelihood = self.app.predict_loglikelihood(sequences)

        scores = [{self.score_names()[0]: ll} for ll in loglikelihood]

        return scores

"""Template file to develop personal scorer
WARNINGS: if you run the scorer locally and don't have a GPU
          you should choose device='cpu'
"""


from pathlib import Path
from typing import Dict, List, Optional

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

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.app = TransformersApp(device=device)

        # FILL IN
        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = None

        # Use load_model for tensorflow/keras model
        # Use load for pytorch model
        if self._checkpoint_filename is not None:
            self.model = load_model(self.get_checkpoint_path())

    def get_checkpoint_path(self) -> str:
        """Return solve checkpoint model path"""
        checkpoint_dir = (Path(__file__).parent / "../checkpoint").resolve()
        return str(checkpoint_dir / self._checkpoint_filename)

    @staticmethod
    def score_names() -> List[str]:
        """
        Criteria names. Must be specified

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

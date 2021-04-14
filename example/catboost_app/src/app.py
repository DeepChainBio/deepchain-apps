"""Template file to develop personal app
WARNINGS: if you run the app locally and don't have a GPU
          you should choose device='cpu'
"""


from pathlib import Path
from typing import Dict, List, Optional

from deepchain.components import DeepChainApp, Transformers
from tensorflow.keras.models import load_model
from catboost import CatBoostRegressor, Pool
import numpy as np

Score = Dict[str, float]
ScoreList = List[Score]

WITH_BIGRAM = True
WITH_BIGRAM_O1 = True

class CatboostAntibodyApp(DeepChainApp):
    """
    DeepChain App template:
    Implement score_names() and compute_score() methods.
    Choose a a transformer available on DeepChain
    Choose a personal keras/tensorflow model
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device

        self.model = CatBoostRegressor(iterations=4000, score_function='L2',
                                                depth=9, min_data_in_leaf=1,
                                                learning_rate=0.04, 
                                                loss_function='RMSE')
        # create dictionary to map Amino acids to integer
        self.i2s = [
            'PAD', 'EOS', 'P', 'V', 'I', 'K', 'N', 'B', 'F', 'Y', 'E', 'W', 'R', 
            'D', 'X', 'S', 'C', 'U', 'Q', 'A', 'M', 'H', 'L', 'G', 'T'
        ]
        self.s2i = { k:v for k , v in zip(self.i2s, range(0,len(self.i2s)))}
        # FIRST axis is the FROM part, 2nd axis is the 2nd item in bigram
        self.i2i_bigram = np.arange(len(self.i2s)**2).reshape((len(self.i2s), len(self.i2s))) 

        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = "catboost_fs12.cat"

        if self._checkpoint_filename is not None:
            self.model.load_model(self.get_checkpoint_path(__file__))

    @staticmethod
    def score_names() -> List[str]:
        """
        App Score Names. Must be specified
        Example:
         return ["max_probability", "min_probability"]
        """
        return ["binding_energy"]

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
        if not isinstance(sequences, list):
            sequences = [sequences]

        for seq in sequences:
            print(len(seq))
            if len(seq) != 221: raise AssertionError("Only sequences of length 221 allowed for this model")

        test_data = np.stack([self._preprocess_seq(s) for s in sequences], axis=0)
        test_pool = Pool(test_data, cat_features=np.arange(221 + (221 if WITH_BIGRAM else 0) + (221 if WITH_BIGRAM_O1 else 0)))
        test_preds = self.model.predict(test_pool)

        pred_list = [{self.score_names()[0]: pred} for pred in test_preds]

        return pred_list

    def _preprocess_seq(self, sequence: str) -> np.ndarray:
        full_cdr = list(sequence)
        full_cdr = [self.s2i[c] for c in full_cdr]
        
        if WITH_BIGRAM:
            fr = np.array(full_cdr)
            fr_r = np.roll(fr, 1)
            bigrams = self.i2i_bigram[fr, fr_r].tolist()
        else: bigrams = []
        if WITH_BIGRAM_O1:
            fr_r2 = np.roll(fr, 2)
            bigrams_o1 = self.i2i_bigram[fr, fr_r2].tolist()
        else: bigrams_o1 = []
        
        inference_item = np.array(full_cdr + bigrams + bigrams_o1)
        return inference_item

if __name__ == "__main__":

    sequences = [
        ("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        )[:221],
        ("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        )[:221],
    ]
    app = App("cpu")
    scores = app.compute_scores(sequences)
    print(scores)

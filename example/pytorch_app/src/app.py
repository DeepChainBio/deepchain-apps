"""
PyTorch implementaton of binding energy scorer
The model scores the binding energy per position in the receptor binding domain of proteins.
By Matthew Baas and Kevin Eloff
"""


from pathlib import Path
from typing import Dict, List, Optional

from deepchain.components import DeepChainApp, Transformers
from tensorflow.keras.models import load_model
import numpy as np

import torch
import torch.nn as nn

from os import path
import urllib

Score = Dict[str, float]
ScoreList = List[Score]

WITH_BIGRAM = True
WITH_BIGRAM_O1 = True

class App(DeepChainApp):
    """
    DeepChain App:
    PyTorch antibody app implementation
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device

        self.model = Model().to(self._device)

        # create dictionary to map Amino acids to integer
        self.i2s = [
            'PAD', 'EOS', 'P', 'V', 'I', 'K', 'N', 'B', 'F', 'Y', 'E', 'W', 'R', 
            'D', 'X', 'S', 'C', 'U', 'Q', 'A', 'M', 'H', 'L', 'G', 'T'
        ]
        self.s2i = { k:v for k , v in zip(self.i2s, range(0,len(self.i2s)))}
        # FIRST axis is the FROM part, 2nd axis is the 2nd item in bigram
        self.i2i_bigram = np.arange(len(self.i2s)**2).reshape((len(self.i2s), len(self.i2s))) 

        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = "model.pth"

        if self._checkpoint_filename is not None:
            self.model.load_state_dict(torch.load(self.get_checkpoint_path(__file__)))

    @staticmethod
    def score_names() -> List[str]:
        """
        Return a list of app score names
        """
        return ["binding energy"]

    def compute_scores(self, sequences: List[str]) -> ScoreList:
        """
        Return a list of all proteins score
        Score is a list of dict:
                - element of list is protein score
                - key of dict are score_names
        sequence must be 221 length proteins
        """
        if not isinstance(sequences, list):
            sequences = [sequences]

        test_data = [self._preprocess_seq(s) for s in sequences]
        test_preds = []
        for item in test_data:
            with torch.no_grad(): test_preds.append(self.model(torch.tensor(item[None], device=self._device)).item())

        pred_list = [{self.score_names()[0]: pred} for pred in test_preds]

        return pred_list

    def _preprocess_seq(self, sequence: str) -> np.ndarray:
        # Amino acids as IDs
        full_cdr = list(sequence)
        full_cdr = [self.s2i[c] for c in full_cdr]
        
        # Amino acid bigrams as IDs
        fr = np.array(full_cdr)
        fr_r = np.roll(fr, 1) # shift sequence by 1
        bigrams = self.i2i_bigram[fr, fr_r].tolist() # combine with shifted sequence 
        
        # Offset amino acids as bigrams
        fr_r2 = np.roll(fr, 2) # shift sequence by 2
        bigrams_o1 = self.i2i_bigram[fr, fr_r2].tolist() # combine with double shifted sequence
        
        inference_item = np.array([
            full_cdr, 
            bigrams, 
            bigrams_o1
        ])
        return inference_item

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.embed_plain = nn.Embedding(25, 128)
        self.embed_bigram = nn.Embedding(25**2, 128)
        self.embed_o_bigram = nn.Embedding(25**2, 128)
        
        self.BiLSTM = nn.LSTM(128*3, 256, num_layers=3, batch_first=True, bidirectional=True)

        self.conv_full = nn.Sequential(nn.Conv1d(128*3, 128, 3, stride=1),
                                       nn.SELU(),
                                       nn.Conv1d(128, 128, 3, stride=1),
                                       nn.SELU(),
                                       nn.Conv1d(128, 256, 3, stride=1),
                                       nn.SELU())
        
        self.hidden_linear = nn.Sequential(nn.Linear(256*2+256, 64), nn.SELU())
        self.out_linear = nn.Linear(64, 1)

    def forward(self, x):
        embed = torch.cat((
            self.embed_plain(x[:,0]), 
            self.embed_bigram(x[:,1]), 
            self.embed_o_bigram(x[:,2])
        ), dim=-1)
                                           
        xn, (hn, cn) = self.BiLSTM(embed)
        cn = torch.max(self.conv_full(embed.transpose(-1,-2)), dim=-1)[0]
        
        conc = torch.cat((xn[:,-1], cn), dim=1)
        out = self.out_linear(self.hidden_linear(conc))
        return out

if __name__ == "__main__":

    sequences = [
        ("QVMLKESGPGLVAPSGSLSITCTVLGFLLDSNGVHWVRQPPGKGLEWLGVIWAGGNTNYNSALMSRVSISKDNSKAQVFLKMKSLQTDDTANYYCARDFYAYDYFYYAMDYWGQGTSVTVSSAFTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVP"),
        ("QVQLKESGPGLVAPQQSLSITCTVSGFLLGTNGVHWVRQPPGKGLEWLGVIWAGGISNYNSALMSRVSISKDNSKSQVFLNMKSLQTDDTAMYYCARDFYDYDVFYYAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVP"),
    ]
    app = App()
    scores = app.compute_scores(sequences)
    print(scores)

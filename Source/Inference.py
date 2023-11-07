from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[0])
sys.path.append(path_git)
from Source.ModelArgs import *
from Source.model.Model import *


from sentencepiece import SentencePieceProcessor

class LlaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, args: ModelArgs):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.args = args


    @staticmethod
    def build()
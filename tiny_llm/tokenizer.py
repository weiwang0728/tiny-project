import os
import struct
from sentencepiece import SentencePieceProcessor
from typing import List

TOKENIZER_MODEL = './data/tok4096.model'

class Tokenizer:
    def __init__(self, model_path=None):
        model_path = model_path if model_path  else TOKENIZER_MODEL
        assert os.path.isfile(model_path)

        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()

    def encode(self, s:str, bos:bool, eos:bool)->List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t
    
    def decode(self, t:List[int]):
        return self.sp_model.decode(t)
        
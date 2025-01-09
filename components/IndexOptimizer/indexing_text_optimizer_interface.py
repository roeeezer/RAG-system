from abc import ABC, abstractmethod
from query import Query
from typing import List
from web_text_unit import WebTextUnit
import trankit
import torch

class IndexingTextOptimizerInterface(ABC):
    @abstractmethod
    def optimize_query(self, text: str) -> str:
        pass

    def optimize_document(self, text: str) -> str:
        pass

class NoneIndexOptimizer(IndexingTextOptimizerInterface):

    @abstractmethod
    def optimize_query(self, text: str) -> str:
        return text
    
    def optimize_document(self, text: str) -> str:
        return text
    
class LemmatizerIndexOptimizer(IndexingTextOptimizerInterface):
    def __init__(self):
        self.pipeline = trankit.Pipeline("hebrew")
        # check if gpu is available
        if torch.cuda.is_available():
            print("Using GPU to lemmatize")
        else:
            print("Using CPU to lemmatize")

        
    def optimize_text(self, text: str) -> str:
        if not text:
            return text
        
        # Perform lemmatization using Trankit
        lemmatize_tokens = self.pipeline.lemmatize(text)

        # Construct lemmatized text by iterating over tokens
        lemmatized_text = " ".join(
            token.get('lemma', token['text']) for sentence in lemmatize_tokens['sentences'] for token in sentence['tokens']
        )
    
        return lemmatized_text



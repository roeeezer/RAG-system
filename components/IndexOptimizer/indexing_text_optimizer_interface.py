from abc import ABC, abstractmethod
import trankit
import torch
from typing import List
import re

class IndexingTextOptimizerInterface(ABC):
    @abstractmethod
    def optimize_query(self, lst_text: List[str]) -> List[str]:
        pass

    def optimize_document(self, lst_text: List[str]) -> List[str]:
        pass

class NoneIndexOptimizer(IndexingTextOptimizerInterface):

    @abstractmethod
    def optimize_query(self, lst_text: List[str]) -> List[str]:
        return lst_text
    
    def optimize_document(self, lst_text: List[str]) -> List[str]:
        return lst_text
    
class LemmatizerIndexOptimizer(IndexingTextOptimizerInterface):
    def __init__(self):
        self.pipeline = trankit.Pipeline("hebrew")
        # check if gpu is available
        if torch.cuda.is_available():
            print("Using GPU to lemmatize")
        else:
            print("Using CPU to lemmatize")

        
    def optimize_document(self, lst_text: List[str]) -> List[str]:
        if not lst_text:
            return lst_text
        
        preprocessed_texts = []
        for text in lst_text:
            # Add spaces around punctuation and special characters
            text = re.sub(r"([^\w\s])", r" \1 ", text)  # Add spaces around non-word characters
            text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces to a single space
            # Remmove * : . , " ' from text
            text = re.sub(r"[*:,\"']", "", text)
            preprocessed_texts.append(text.strip().split())
        # Perform lemmatization using Trankit

        lemmatize_tokens = self.pipeline.lemmatize(preprocessed_texts)

        # Construct lemmatized text by iterating over tokens

        lemmatized_texts = [
            " ".join(
                token['text'] if token.get('lemma', "_") == "_" else token['lemma']
                for token in sentence['tokens']
            )
            for sentence in lemmatize_tokens['sentences']
        ]
        return lemmatized_texts
    
    def optimize_query(self, lst_text: List[str]) -> List[str]:
        return self.optimize_document(lst_text)



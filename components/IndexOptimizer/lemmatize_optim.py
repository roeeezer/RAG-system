from typing import List
import trankit
import torch
import re
from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface


def create_lemmatized_sentence(trankit_output: dict) -> str:
    # Extract the first sentence from the output (assuming single sentence input)
    lemmatized_sentences = []

    for sentence in trankit_output.get('sentences', []):
        lemmatized_sentence = " ".join(
            " ".join(expanded_token['lemma'] for expanded_token in token['expanded'])  # Join expanded lemmas
            if 'expanded' in token else token['lemma']  # Use lemma if no expansion
            for token in sentence['tokens']
        )
        lemmatized_sentences.append(lemmatized_sentence)
    
    return lemmatized_sentences


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
        
        lemmatized_texts = []
        for text in lst_text:
            lemmatize_text = create_lemmatized_sentence(self.pipeline.lemmatize(text))
            lemmatized_texts.append(lemmatize_text)


        return lemmatized_texts
    
    def optimize_query(self, lst_text: List[str]) -> List[str]:
        return self.optimize_document(lst_text)
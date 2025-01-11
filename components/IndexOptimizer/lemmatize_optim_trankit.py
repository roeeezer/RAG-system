from typing import List
import trankit
import torch
from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface


def create_lemmatized_text(trankit_output: dict) -> str:
    """
    Converts the Trankit parsed output to a single lemmatized text (not split into sentences).
    """
    lemmatized_text = " ".join(
        " ".join(expanded_token['lemma'] for expanded_token in token['expanded'])  # Join expanded lemmas
        if 'expanded' in token else token['lemma']  # Use lemma if no expansion
        for sentence in trankit_output.get('sentences', [])
        for token in sentence['tokens']
    )
    return lemmatized_text


class LemmatizerIndexOptimizerTrankit(IndexingTextOptimizerInterface):
    def __init__(self):
        self.pipeline = trankit.Pipeline("hebrew")
        if torch.cuda.is_available():
            print("Using GPU to lemmatize")
        else:
            print("Using CPU to lemmatize")
    
    def optimize_documents(self, lst_text: List[str]) -> List[str]:
        """
        Concatenates all texts, lemmatizes as one, and splits the result based on the original grouping.
        """
        if not lst_text:
            return lst_text
        # Concatenate all input texts into one large text
        concatenated_text = " [SEP] ".join(lst_text)  # Use a special marker to track original splits

        # Perform lemmatization
        lemmatized_output = self.pipeline.lemmatize(concatenated_text)
        lemmatized_text = create_lemmatized_text(lemmatized_output)

        # Split the lemmatized text based on the special marker
        lemmatized_texts = lemmatized_text.split("[ SEP ]")


        return lemmatized_texts
    
    def optimize_queries(self, lst_text: List[str]) -> List[str]:
        return self.optimize_documents(lst_text)

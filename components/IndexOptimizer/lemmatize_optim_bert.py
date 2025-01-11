from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface


def make_lemmatized_sentence(bert_output: List[List[tuple]]) -> List[str]:
    """
    Converts BERT output to lemmatized sentences.
    
    Args:
        bert_output (List[List[tuple]]): The BERT output where each inner list contains tuples of (token, lemma).
    
    Returns:
        List[str]: A list of lemmatized sentences.
    """
    lemmatized_sentences = []

    for sentence in bert_output:
        lemmatized_sentence = " ".join(lemma for token, lemma in sentence)  # Join all lemmas for each sentence
        lemmatized_sentences.append(lemmatized_sentence)
    
    return lemmatized_sentences


class LemmatizerIndexOptimizerBert(IndexingTextOptimizerInterface):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-lex")
        self.model = AutoModel.from_pretrained("dicta-il/dictabert-lex", trust_remote_code=True)
        self.model.to(self.device)
    
    def optimize_documents(self, lst_text: List[str]) -> List[str]:
        outputs = self.model.predict(lst_text, self.tokenizer)
        return make_lemmatized_sentence(outputs)

    def optimize_queries(self, lst_text: List[str]) -> List[str]:
        return self.optimize_documents(lst_text)

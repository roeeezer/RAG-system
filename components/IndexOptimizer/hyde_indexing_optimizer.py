from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface
from components.LlmAnswerRetriever.gemini import Gemini
from typing import List

class HydeIndexingOptimizer(IndexingTextOptimizerInterface):

    def __init__(self, gemini : Gemini):
        self.gemini = gemini

    def optimize_query(self, lst_text: List[str]) -> List[str]:
        res = []
        for text in lst_text:
            res.append(self.gemini.get_llm_output(text))
        return res

    def optimize_document(self, lst_text: List[str]) -> List[str]:
        return lst_text
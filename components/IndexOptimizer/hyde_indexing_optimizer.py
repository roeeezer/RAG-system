from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface
from components.LlmAnswerRetriever.gemini import Gemini
from typing import List

class HydeIndexingOptimizer(IndexingTextOptimizerInterface):

    def __init__(self, gemini : Gemini):
        self.gemini = gemini

    def optimize_queries(self, queries: List[str]) -> List[str]:
        res = []
        for query in queries:
            res.append(query + " " + self.gemini.get_llm_output(query))
        return res

    def optimize_documents(self, lst_text: List[str]) -> List[str]:
        return lst_text
from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface
from components.LlmAnswerRetriever.gemini import Gemini

class HydeIndexingOptimizer(IndexingTextOptimizerInterface):

    def __init__(self, gemini : Gemini):
        self.gemini = gemini

    def optimize_query(self, text: str) -> str:
        return self.gemini.get_llm_output(text) 

    def optimize_document(self, text: str) -> str:
        return text
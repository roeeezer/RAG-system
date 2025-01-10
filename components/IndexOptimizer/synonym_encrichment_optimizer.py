from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface
from components.hebrew_synonym_expander import HebrewSynonymExpander

class SynonymEnrichmentOptimizer(IndexingTextOptimizerInterface):

    def __init__(self, top_k: int = 5):
        self.expander = HebrewSynonymExpander(top_k=top_k)
    
    def optimize_query(self, lst_text: list[str]) -> list[str]:
        res = []
        for query in lst_text:
            expanded = self.expander.expand_query(query)
            res.append(expanded)
        return res

    def optimize_document(self, lst_text: list[str]) -> list[str]:
        return lst_text
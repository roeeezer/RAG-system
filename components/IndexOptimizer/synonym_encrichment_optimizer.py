import os
import pickle
from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface
from components.SynonymExpanders.hebrew_synonym_expander import HebrewSynonymExpander
from components.web_text_unit import WebTextSection


class SynonymEnrichmentOptimizer(IndexingTextOptimizerInterface):

    def __init__(self, top_k: int):
        self.top_k = top_k
        self.expander = HebrewSynonymExpander(top_k=self.top_k)
        self.cache_file = None
    
    def optimize_queries(self, lst_text: list[str]) -> list[str]:
        self.cache_file = os.path.join("cache","synonym optimizer", f"e_{self.expander.__class__.__name__}k_{self.top_k} q_{len(lst_text)}.pkl")
        try:
            res = self._load_from_cache()
            return res
        except FileNotFoundError:
            res = []
            for query in lst_text:
                expanded = self.expander.expand_query(query)
                res.append(expanded)
            self._save_to_cache(res)
            return res

    def optimize_documents(self, lst_text: list[str]) -> list[str]:
        return lst_text
    
    def _save_to_cache(self, data: list[WebTextSection]):
        """Shared method to save processed data to pickle file"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)

    def _load_from_cache(self) -> list[WebTextSection]:
        """Shared method to load processed data from pickle file"""
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)
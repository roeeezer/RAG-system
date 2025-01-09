from abc import ABC, abstractmethod
from components.query import Query
from typing import List
from components.web_text_unit import WebTextUnit

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
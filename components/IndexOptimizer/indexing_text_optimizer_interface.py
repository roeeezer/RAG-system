from abc import ABC, abstractmethod
from components.query import Query
from typing import List
from components.web_text_unit import WebTextUnit

class IndexingTextOptimizerInterface(ABC):
    @abstractmethod
    def optimize_text(self, text: str) -> str:
        pass

class NoneIndexOptimizer(IndexingTextOptimizerInterface):

    def optimize_text(self, text: str) -> str:
        return text
from abc import ABC, abstractmethod
import trankit
import torch
from typing import List
import re


class IndexingTextOptimizerInterface(ABC):
    @abstractmethod
    def optimize_queries(self, queries: List[str]) -> List[str]:
        pass

    def optimize_documents(self, documents: List[str]) -> List[str]:
        pass

class NoneIndexOptimizer(IndexingTextOptimizerInterface):

    @abstractmethod
    def optimize_queries(self, lst_text: List[str]) -> List[str]:
        return lst_text
    
    def optimize_documents(self, lst_text: List[str]) -> List[str]:
        return lst_text
    




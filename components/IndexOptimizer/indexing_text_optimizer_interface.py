from abc import ABC, abstractmethod
import trankit
import torch
from typing import List
import re


class IndexingTextOptimizerInterface(ABC):
    @abstractmethod
    def optimize_query(self, lst_text: List[str]) -> List[str]:
        pass

    def optimize_document(self, lst_text: List[str]) -> List[str]:
        pass

class NoneIndexOptimizer(IndexingTextOptimizerInterface):

    @abstractmethod
    def optimize_query(self, lst_text: List[str]) -> List[str]:
        return lst_text
    
    def optimize_document(self, lst_text: List[str]) -> List[str]:
        return lst_text
    




from abc import ABC, abstractmethod

class WebTextUnit(ABC):
    @abstractmethod
    def get_id(self) -> str:
        pass

    @abstractmethod
    def get_text(self) -> str:
        pass

class WebTextSection:

    def __init__(self, doc_id, section_id, text):
        self.doc_id = doc_id
        self.section_id = section_id
        self.text = text

    def get_id(self) -> str:
        return self.doc_id + "_" + self.section_id

    def get_text(self) -> str:
        return self.text
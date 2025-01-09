from abc import ABC, abstractmethod
from dataclasses import dataclass

class WebTextUnit(ABC):
    @abstractmethod
    def get_id(self) -> str:
        pass

    @abstractmethod
    def get_content(self) -> str:
        pass

    @abstractmethod
    def get_lemmatised_content(self) -> str:
        pass

@dataclass
class WebTextSection:
    doc_id: str
    section_id: str
    content: str
    lemmatised_content: str

    def get_id(self) -> str:
        return self.doc_id + "_" + self.section_id

    def get_content(self) -> str:
        return self.content
    
    def get_lemmatised_content(self) -> str:
        return self.lemmatised_content

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "text": self.content,
            "lemma" : self.lemmatised_content
        }
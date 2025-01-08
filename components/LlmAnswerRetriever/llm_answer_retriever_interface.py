from abc import ABC, abstractmethod
from components.web_text_unit import WebTextSection
from components.query import Query

def reverse_lines(paragraph):
   # Split to lines, reverse each line's chars, rejoin with newlines
   lines = paragraph.split('\n')
   reversed_lines = [''.join(reversed(line)) for line in lines]
   return '\n'.join(reversed_lines)

class LlmAnswerRetrieverInterface(ABC):
    @abstractmethod
    def retrieve_final_answers(self, queries: list[Query]):
        pass

class EmptyAnswerRetrieverInterface(LlmAnswerRetrieverInterface):
    def retrieve_final_answers(self, queries: list[Query]):
        return



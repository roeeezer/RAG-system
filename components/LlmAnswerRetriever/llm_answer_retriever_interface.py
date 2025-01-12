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

    @abstractmethod
    def get_sent_tokens_counter(self) -> int:
        pass    

class EmptyAnswerRetrieverInterface(LlmAnswerRetrieverInterface):
    def __init__(self):
        self.sent_tokens_counter = 0

    def retrieve_final_answers(self, queries: list[Query]):
        for query in queries:
            query_counter = 0
            query_counter += len(query.query.split())
            for answer_source in query.answer_sources:
                query_counter += len(answer_source.get_content().split())
            self.sent_tokens_counter += query_counter
        return
    
    def get_sent_tokens_counter(self) -> int:
        return self.sent_tokens_counter   



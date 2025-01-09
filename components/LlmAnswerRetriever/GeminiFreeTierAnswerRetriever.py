import os
from components.query import Query
from tqdm import tqdm
from components.LlmAnswerRetriever.llm_answer_retriever_interface import LlmAnswerRetrieverInterface
from components.LlmAnswerRetriever.gemini import Gemini

class GeminiFreeTierAnswerRetriever(LlmAnswerRetrieverInterface):
    def __init__(self, gemini: Gemini):
        self.gemini = gemini

    def get_llm_output(self, llm_input):
        return self.gemini.get_llm_output(llm_input)
    
    def get_llm_input(self, query, answer_source):
        script_directory = os.path.dirname(__file__)
        file_path = os.path.join(script_directory, "llm_input_pattern.txt")
        with open(file_path, encoding="utf-8") as file:
            pattern = file.read()
        
        filled_pattern = pattern.replace("{Query}", query).replace("{AnswerSource}", answer_source)
        return filled_pattern

    def retrieve_final_answers(self, queries: list[Query]):
        for query in tqdm(queries, desc="Retrieving final answers from Gemini"):
            if len(query.answer_sources) > 0:
                llm_input = self.get_llm_input(query.query, " ".join([source.get_content() for source in query.answer_sources]))
                llm_output = self.get_llm_output(llm_input)
                query.final_answer = llm_output
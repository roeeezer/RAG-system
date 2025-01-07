import os
from abc import ABC, abstractmethod
import google.generativeai as genai
from components.web_text_unit import WebTextSection
from components.query import Query
import time
from tqdm import tqdm

def reverse_lines(paragraph):
   # Split to lines, reverse each line's chars, rejoin with newlines
   lines = paragraph.split('\n')
   reversed_lines = [''.join(reversed(line)) for line in lines]
   return '\n'.join(reversed_lines)

class LlmAnswerRetrieverInterface(ABC):
    @abstractmethod
    def retrieve_final_answers(self, queries: list[Query]):
        pass

class GeminiFreeTierAnswerRetriever(LlmAnswerRetrieverInterface):
    def get_api_key(self):
        current_directory_path = os.getcwd()
        file_path = os.path.join(current_directory_path, "untracked", "gemini_api_key.txt")
        api_key = open(file_path, encoding='utf-8').read()
        return api_key

    def get_llm_output(self, llm_input):
        api_key = self.get_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        retries = 2
        for attempt in range(retries + 1):
            try:
                response = model.generate_content(llm_input)
                return response.text
            except Exception as e:
                if attempt < retries:
                    print(f"Request failed: {e}. Retrying in 70 seconds...")
                    time.sleep(70)
                else:
                    print(f"Request failed after {retries + 1} attempts: {e}")
                    raise
    
    def get_llm_input(self, query, answer_source):
        script_directory = os.path.dirname(__file__)
        file_path = os.path.join(script_directory, "llm_input_pattern.txt")
        with open(file_path, encoding="utf-8") as file:
            pattern = file.read()
        
        filled_pattern = pattern.replace("{Query}", query).replace("{AnswerSource}", answer_source)
        return filled_pattern

    def retrieve_final_answers(self, queries: list[Query]):
        for query in tqdm(queries, desc="Retrieving final answers from Gemini"):
            if len(query.answer_source) > 0:
                llm_input = self.get_llm_input(query.query, " ".join([source.get_content() for source in query.answer_source]))
                llm_output = self.get_llm_output(llm_input)
                query.final_answer = llm_output

import os
from abc import ABC, abstractmethod
import google.generativeai as genai
from components.web_text_unit import WebTextSection

def reverse_lines(paragraph):
   # Split to lines, reverse each line's chars, rejoin with newlines
   lines = paragraph.split('\n')
   reversed_lines = [''.join(reversed(line)) for line in lines]
   return '\n'.join(reversed_lines)

class GetFinalAnswersInterface(ABC):
    @abstractmethod
    def get_final_answers(self, queries: list[str], answer_sources : list[WebTextSection]):
        pass

class GetFinalAnswersImplementation(GetFinalAnswersInterface):
    def get_api_key(self):
        current_directory_path = os.getcwd()
        file_path = os.path.join(current_directory_path, "untracked", "gemini_api_key.txt")
        api_key = open(file_path, encoding='utf-8').read()
        return api_key

    def get_llm_output(self, llm_input):
        api_key = self.get_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(llm_input)
        return response.text
    
    def get_llm_input(self, query, answer_source):
        script_directory = os.path.dirname(__file__)
        file_path = os.path.join(script_directory, "llm_input_pattern.txt")
        with open(file_path, encoding="utf-8") as file:
            pattern = file.read()
        
        filled_pattern = pattern.replace("{Query}", query).replace("{AnswerSource}", answer_source)
        print(reverse_lines(filled_pattern))
        return filled_pattern

    def get_final_answers(self, queries: list[str], answer_sources : list[WebTextSection]):
        final_answers = []
        if len(queries) != len(answer_sources):
            raise Exception("Queries and answer_sources must have the same length.")
        for i in range(len(queries)):
            llm_input = self.get_llm_input(queries[i], answer_sources[i][0].get_text())
            llm_output = self.get_llm_output(llm_input)
            final_answers.append(llm_output)
        return final_answers

import os
import google.generativeai as genai
import time
from components.LlmAnswerRetriever.llm_answer_retriever_interface import LlmAnswerRetrieverInterface

class Gemini():
        
    def get_api_key(self):
        current_directory_path = os.getcwd()
        file_path = os.path.join(current_directory_path, "untracked", "gemini_api_key.txt")
        api_key = open(file_path, encoding='utf-8').read()
        return api_key
    
    def __init__(self):
        self.api_key = self.get_api_key()
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.wait_time = 61

    def get_llm_output(self, llm_input):
        retries = 2
        for attempt in range(retries + 1):
            try:
                response = self.model.generate_content(llm_input)
                return response.text
            except Exception as e:
                if attempt < retries:
                    print(f"Request failed: {e}. Retrying in 70 seconds...")
                    time.sleep(self.wait_time)
                else:
                    print(f"Request failed after {retries + 1} attempts: {e}")
                    raise
import os
import google.generativeai as genai
import time
from components.LlmAnswerRetriever.llm_answer_retriever_interface import LlmAnswerRetrieverInterface
import time


class Gemini():
    
    def get_api_keys(self):
        current_directory_path = os.getcwd()
        file_path = os.path.join(current_directory_path, "untracked", "gemini_api_keys.txt")
        with open(file_path, encoding='utf-8') as f:
            api_keys = [line.strip() for line in f if line.strip()]
        return api_keys
    
    def __init__(self, constraint_model=False):
        self.api_keys = self.get_api_keys()
        self.current_key_index = 0
        self.last_exception_time = None
        self.wait_time = 61
        self.set_api_key(self.api_keys[self.current_key_index])
        self.tokens_counter = 0
        self.constraint_model = constraint_model

    def set_api_key(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def get_llm_output(self, llm_input):
        input_tokens = llm_input.split()
        if self.constraint_model:
            if len(input_tokens) > 1000:
                # print("Tokens limit exceeded")
                # print("llm_input[:100]: ", llm_input[:100]) 
                input_tokens = input_tokens[:1000]
                llm_input = " ".join(input_tokens)
        retries = len(self.api_keys)
        for attempt in range(retries):
            try:
                response = self.model.generate_content(llm_input)
                self.tokens_counter += len(input_tokens)
                return response.text
            except Exception as e:
                print(f"Request failed with API key {self.current_key_index}: {e}")
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                self.set_api_key(self.api_keys[self.current_key_index])
                
                if self.last_exception_time is None:
                    self.last_exception_time = time.time()
                
                elapsed_time = time.time() - self.last_exception_time
                if self.current_key_index == 0 and elapsed_time < self.wait_time:
                    wait_time_remaining = self.wait_time - elapsed_time
                    print(f"All keys used. Waiting for {wait_time_remaining:.2f} seconds...")
                    time.sleep(wait_time_remaining)
                    self.last_exception_time = None

                if attempt == retries - 1:
                    print(f"Request failed after {retries} attempts: {e}")
                    raise

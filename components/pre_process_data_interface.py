from abc import ABC, abstractmethod
from bs4 import BeautifulSoup, Tag
import glob
import re
import torch
from markdownify import MarkdownConverter
from components.web_text_unit import WebTextUnit, WebTextSection
from tqdm import tqdm
import pickle
import time
from typing import List
import os
import trankit
from tqdm import tqdm

debug_mode = False

def debug_print(*args, **kwargs):
    if debug_mode:
        print(*args, **kwargs)

def reverse_lines(paragraph):
   # Split to lines, reverse each line's chars, rejoin with newlines
   lines = paragraph.split('\n')
   reversed_lines = [''.join(reversed(line)) for line in lines]
   return '\n'.join(reversed_lines)

class CustomConverter(MarkdownConverter):
    # don't format markdown links, return only their text, not their URL
    def convert_a(self, element, text, convert_as_inline):
        """Override link conversion to return only text without URL"""
        return text

    # skip tables
    def convert_table(self, element, text, convert_as_inline):
        """Skip table formatting, replace with placeholder"""
        return "[טבלה]"

class PreProcessDataInterface(ABC):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.cache_file = os.path.join("cache", f"{self.data_path}.pkl")

    @abstractmethod
    def pre_proccess_data(self) -> list[WebTextUnit]:
        """Each implementation must provide its own data processing logic"""
        pass

    def load_or_process_data(self) -> List[WebTextSection]:
        """
        Shared caching logic for all implementations.
        Attempts to load preprocessed data from cache, falls back to processing files
        if cache doesn't exist.
        """
        try:
            start_time = time.time()
            data = self._load_from_cache()
            load_time = time.time() - start_time
            print(f"Loaded cached data in {load_time:.2f} seconds")
            return data
        except FileNotFoundError:
            start_time = time.time()
            data = self.pre_proccess_data()  # Calls the implementation-specific method
            process_time = time.time() - start_time
            
            # Save to cache
            save_start = time.time()
            self._save_to_cache(data)
            save_time = time.time() - save_start
            
            print(f"Processed files in {process_time:.2f} seconds")
            print(f"Saved cache in {save_time:.2f} seconds")
            
            return data

    def _save_to_cache(self, data: List[WebTextSection]):
        """Shared method to save processed data to pickle file"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)

    def _load_from_cache(self) -> List[WebTextSection]:
        """Shared method to load processed data from pickle file"""
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)
    
class WebDataPreProccessor(PreProcessDataInterface):
    def pre_proccess_data(self) -> list[WebTextUnit]:
        res = []
        html_files = glob.glob(f"{self.data_path}/pages/*.html")
        for index, html_file_path in enumerate(tqdm(html_files, desc="Processing HTML files")):
            with open(html_file_path, encoding='utf-8') as file:
                html_content = BeautifulSoup(file.read(), "html.parser")
            document_id = html_file_path.split("/")[-1].replace(".html", "").replace("pages\\", "")
            page_title = html_content.title.contents[0]
            
            # Get main content section
            main_content = html_content.main
            
            # Convert HTML to Markdown format
            markdown_content = CustomConverter(heading_style="ATX", bullets="*").convert_soup(main_content)
            
            # Clean up excessive newlines
            markdown_content = re.sub(r"\n\n+", "\n\n", markdown_content)
            
            # Split content into sections based on headers
            content_sections = markdown_content.split("\n#")
            
            # Process each section
            for i, section_content in enumerate(content_sections):
                if not section_content.strip():
                    continue  # Skip empty sections
                    
                # Restore header marker and split into title and body
                section_content = "#" + section_content
                section_title, section_body = section_content.split("\n", 1)
                
                res.append(WebTextSection(document_id, str(i), page_title + section_title + section_body, None))
        return res
    

class WebDataPreProccessorLemmatization(PreProcessDataInterface):
    def __init__(self, data_path: str):
        super().__init__(data_path)
        # Initialize Trankit pipeline
        self.pipeline = trankit.Pipeline(lang='hebrew', gpu=True)
        # PyTorch version
        print(f"PyTorch version: {torch.__version__}")

        # Check if CUDA is available (GPU support)
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Check CUDA version PyTorch was built with
        print(f"Built with CUDA version: {torch.version.cuda}")

        # If GPU is available, print GPU device details
        if torch.cuda.is_available():
            print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("No GPU detected, using CPU.")



    def pre_proccess_data(self) -> list[WebTextUnit]:
        print("Preprocessing data with Trankit...")
        res = []
        html_files = glob.glob(f"{self.data_path}/pages/*.html")
        if len(html_files) == 0:
            raise FileNotFoundError(f"No HTML files found in {self.data_path}/pages")
        if debug_mode:
            files_iter = html_files
        else:
            files_iter = tqdm(html_files, desc="Processing HTML files")


        for index, html_file_path in enumerate(files_iter):
            with open(html_file_path, encoding='utf-8') as file:
                html_content = BeautifulSoup(file.read(), "html.parser")
            # throw error if no content
            if html_content is None:
                raise ValueError(f"Could not parse HTML file: {html_file_path}")

            document_id = html_file_path.split("/")[-1].replace(".html", "").replace("pages\\", "")
            page_title = html_content.title.contents[0]
            debug_print(f"######### document_id:{document_id} #########")
            debug_print(f"Title:\n {reverse_lines(page_title)}")

            # Get main content section
            main_content = html_content.main
            
            # Convert HTML to Markdown format
            markdown_content = CustomConverter(heading_style="ATX", bullets="*").convert_soup(main_content)
            
            # Clean up excessive newlines
            markdown_content = re.sub(r"\n\n+", "\n\n", markdown_content)
            
            # Split content into sections based on headers
            content_sections = markdown_content.split("\n#")
            
            # Process each section
            for i, section_content in enumerate(tqdm(content_sections)):
                if not section_content.strip():
                    continue  # Skip empty sections

                # Restore header marker and split into title and body
                section_content = section_content
                section_title, section_body = section_content.split("\n", 1)
                if "גורמים מסייעים" in section_title:
                    continue
                debug_print(f"### section {i} ###")
                debug_print(f"section_title:\n {reverse_lines(section_title)}")
                debug_print(f"section_body:\n {reverse_lines(section_body)}")

                # Apply Trankit preprocessing
                lemmatized_text = self._preprocess_with_trankit(section_body)

                res.append(WebTextSection(document_id, str(i), page_title + section_title + section_body, lemmatized_text))

        return res

    def _preprocess_with_trankit(self, text: str) -> str:
        """Use Trankit to tokenize and lemmatize the input text"""
        if not text:
            return text
        
        # Perform lemmatization using Trankit
        lemmatize_tokens = self.pipeline.lemmatize(text)

        # Construct lemmatized text by iterating over tokens
        lemmatized_text = " ".join(
            token.get('lemma', token['text']) for sentence in lemmatize_tokens['sentences'] for token in sentence['tokens']
        )
    
        return lemmatized_text

    
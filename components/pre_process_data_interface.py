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

            # **Batch processing: Split sections into batches for Trankit**
            batch_size = 10  # You can adjust this based on available memory and performance
            for i in range(0, len(content_sections), batch_size):
                batch = content_sections[i:i + batch_size]
                res.extend(self._process_section_batch(document_id, page_title, batch, i))

        return res

    def _process_section_batch(self, document_id: str, page_title: str, sections: list[str], start_index: int) -> list[WebTextUnit]:
        """Processes a batch of text sections"""
        valid_sections = []
        for i, section_content in enumerate(sections):
            if not section_content.strip():
                continue  # Skip empty sections

            # Restore header marker and split into title and body
            section_content = section_content
            try:
                section_title, section_body = section_content.split("\n", 1)
            except ValueError:
                section_title, section_body = section_content, ""
            
            if "גורמים מסייעים" in section_title:
                continue

            debug_print(f"### section {start_index + i} ###")
            debug_print(f"section_title:\n {reverse_lines(section_title)}")
            debug_print(f"section_body:\n {reverse_lines(section_body)}")

            valid_sections.append((section_title, section_body))

        # **Batch lemmatization with Trankit**
        section_texts = [section_body for _, section_body in valid_sections]
        lemmatized_texts = self._preprocess_with_trankit(section_texts)

        # Create WebTextSection objects
        result = [
            WebTextSection(
                document_id,
                str(start_index + i),
                page_title + section_title + section_body,
                lemmatized_text
            )
            for i, ((section_title, section_body), lemmatized_text) in enumerate(zip(valid_sections, lemmatized_texts))
        ]
        return result



    def _preprocess_with_trankit(self, texts: list[str]) -> list[str]:
        """Use Trankit to tokenize and lemmatize a batch of text sections with punctuation splitting."""
        if not texts:
            return []

        # Preprocess: split punctuation (e.g., "200(ג)" to "200 , ג")
        preprocessed_texts = []
        for text in texts:
            # Add spaces around punctuation and special characters
            text = re.sub(r"([^\w\s])", r" \1 ", text)  # Add spaces around non-word characters
            text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces to a single space
            preprocessed_texts.append(text.strip())

        # Convert each text section into a list of sentences (list of lists of strings)
        batch_input = [[sentence for sentence in text.split(" ") if sentence.strip()] for text in preprocessed_texts]
        batch_input = [section for section in batch_input if section]

        # Check for empty lists in batch_input
        empty_indices = [i for i, section in enumerate(batch_input) if not section]
        if empty_indices:
            raise ValueError(f"Empty section detected at indices: {empty_indices}. Please check your input data.")

        # Perform lemmatization using Trankit (batch)
        lemmatize_result = self.pipeline.lemmatize(batch_input)

        # Process the lemmatization results
        lemmatized_texts = []
        for section_sentences in lemmatize_result["sentences"]:
            lemmatized_text = " ".join(
                token['text'] if token.get('lemma', "_") == "_" else token['lemma']
                for token in section_sentences["tokens"]
            )
            lemmatized_texts.append(lemmatized_text)



        return lemmatized_texts







    
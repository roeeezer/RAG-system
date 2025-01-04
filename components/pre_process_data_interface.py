from abc import ABC, abstractmethod
from bs4 import BeautifulSoup, Tag
import glob
import re
from markdownify import MarkdownConverter
from components.web_text_unit import WebTextUnit, WebTextSection

class PreProcessDataInterface(ABC):
    @abstractmethod
    def pre_proccess_data(self) -> list[WebTextUnit]:
        pass

class CustomConverter(MarkdownConverter):
    # don't format markdown links, return only their text, not their URL
    def convert_a(self, element, text, convert_as_inline):
        """Override link conversion to return only text without URL"""
        return text

    # skip tables
    def convert_table(self, element, text, convert_as_inline):
        """Skip table formatting, replace with placeholder"""
        return "[טבלה]"
    
class WebDataPreProccessor(PreProcessDataInterface):
    def pre_proccess_data(self) -> list[WebTextUnit]:
        res = []
        for index, html_file_path in enumerate(glob.iglob("created_kol_zchut_corpus_small/pages/*.html")):
            html_content = BeautifulSoup(open(html_file_path, encoding='utf-8').read(), "html.parser")
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
                
                res.append(WebTextSection(document_id, i, page_title + section_title + section_body))
        return res
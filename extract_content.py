from bs4 import BeautifulSoup, Tag
import glob
import re
from markdownify import MarkdownConverter

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

# BeautifulSoup is a very strong package that allows you to target and extract very specific parts of the page,
# using tag-names, ids, css-selectors, and more. It is worth reading its documentation. Here, we use it very basically:
# - we extract the "title" and the "main" tags in the document
# - we then pass "main" to a custom MarkdownConverter to convert its content to markdown, while skipping table formatting and formatting links as text only.
# - We then manually split the markdown string into sections.
# You can do much more with BeautifulSoup and its worth looking at its documentation.
# You probably also want to not print everything to the same unformatted string as we do here, but create some data-structure,
# or save to a structured file (say where each item you care about is a jsonl string) or to multiple files.

for index, html_file_path in enumerate(glob.iglob("created_kol_zchut_corpus/pages/*.html")):
    # Parse HTML content using BeautifulSoup
    html_content = BeautifulSoup(open(html_file_path, encoding='utf-8').read(), "html.parser")
    
    # Extract document ID from filename (remove path and extension)
    document_id = html_file_path.split("/")[-1].replace(".html", "")
    print(f"--------------------------------{document_id}-------------------------------------")
    
    # Extract page title from HTML
    page_title = html_content.title.contents[0]
    print("Title:", reverse_lines(page_title))
    
    # Get main content section
    main_content = html_content.main
    
    # Convert HTML to Markdown format
    markdown_content = CustomConverter(heading_style="ATX", bullets="*").convert_soup(main_content)
    
    # Clean up excessive newlines
    markdown_content = re.sub(r"\n\n+", "\n\n", markdown_content)
    
    # Split content into sections based on headers
    content_sections = markdown_content.split("\n#")
    
    # Process each section
    for section_content in content_sections:
        if not section_content.strip():
            continue  # Skip empty sections
            
        # Restore header marker and split into title and body
        section_content = "#" + section_content
        section_title, section_body = section_content.split("\n", 1)
        
        print(f"SEC--{reverse_lines(section_title)}--------------------")
        print(reverse_lines(section_body))
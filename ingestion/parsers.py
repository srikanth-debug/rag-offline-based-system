
import os
import pandas as pd
from pypdf import PdfReader
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

def parse_pdf_ocr(file_path: str) -> tuple[str, dict]:
    """
    Parses a PDF file, attempting OCR as a fallback for scanned pages.
    Returns the text content and metadata.
    """
    reader = PdfReader(file_path)
    text_content = ""
    is_scanned = False

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        # If text extraction is poor (e.g., < 100 chars), assume it's scanned
        if page_text is None or len(page_text.strip()) < 100:
            is_scanned = True
            try:
                # Convert PDF page to an image
                images = convert_from_path(file_path, first_page=i + 1, last_page=i + 1)
                if images:
                    # Perform OCR on the image
                    page_text = pytesseract.image_to_string(images[0])
            except Exception as e:
                print(f"Warning: OCR failed for page {i+1} in {file_path}: {e}")
                page_text = "" # Assign empty string if OCR fails
        
        text_content += page_text + "\n"

    metadata = {"source": os.path.basename(file_path), "is_scanned": is_scanned}
    return text_content, metadata

def parse_html(file_path: str) -> tuple[str, dict]:
    """Parses an HTML file, extracts text and title."""
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    
    text_content = soup.get_text(separator='\n', strip=True)
    title = soup.title.string if soup.title else "No Title"
    metadata = {"source": os.path.basename(file_path), "title": title}
    return text_content, metadata

def parse_md(file_path: str) -> tuple[str, dict]:
    """Parses a Markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    md = MarkdownIt()
    html_content = md.render(md_content)
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text(separator='\n', strip=True)
    metadata = {"source": os.path.basename(file_path)}
    return text_content, metadata

def parse_csv(file_path: str) -> tuple[str, dict]:
    """Parses a CSV file, converting each row into a text string."""
    df = pd.read_csv(file_path)
    text_content = ""
    for index, row in df.iterrows():
        row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
        text_content += row_text + "\n\n"
    metadata = {"source": os.path.basename(file_path)}
    return text_content, metadata

# The dictionary that maps file extensions to the functions above
PARSER_MAPPING = {
    ".pdf": parse_pdf_ocr,
    ".html": parse_html,
    ".md": parse_md,
    ".csv": parse_csv,
}
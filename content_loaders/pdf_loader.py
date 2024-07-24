"""
Package pre-requisite:
pypdf
pymupdf
langchain
langchain_community
"""

from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
import fitz # installed with pymupdf

def load_pdf_basic(file_path: str) -> List:
    """
    load_pdf_basic uses PyPDF, which is suitable for simpler tasks that involve basic manipulation and extraction from PDFs.
    It is lightweight and easy to use but may struggle with more complex documents.
    use load_pdf_advanced if more complex actions are required.
    """
    print(f"Loading {file_path} ...", end="")
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    print("Done")
    return pages

def load_pdf_advanced(file_path: str) -> List:
    """
    load_pdf_advanced uses PyMuPDF, which offers high-performance rendering and extensive support for complex PDF features.
    Ideal for applications requiring advanced text extraction or image rendering.
    """
    print(f"Loading {file_path} ...", end="")
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    print("Done")
    return data

def pdf_page_to_image(file_path: str, page_number: int) -> None:
    """
    pdf_page_to_image uses fitz, a module inside PyMuPDF to render the pdf page into an image
    This function will output a png image in the root folder
    """
    print(f"Extracting {file_path}, page number {page_number} ...")
    doc = fitz.open(file_path)
    page_number = page_number
    page = doc.load_page(page_number)

    # Define the resolution (pixels per inch)
    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    matrix = fitz.Matrix(zoom_x, zoom_y)

    # Render the page to a pixmap
    pix = page.get_pixmap(matrix=matrix)

    # Save the rendered image to a file
    output_image = f"page_{page_number}.png"
    pix.save(output_image)
    print("Done")
from typing import List
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

def load_docx_basic(file_path: str) -> List:

    print(f"Loading {file_path}...", end="")
    loader = UnstructuredWordDocumentLoader(file_path)
    data = loader.load()
    print("Done")
    return data
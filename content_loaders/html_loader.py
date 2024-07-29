from typing import List
from langchain_community.document_loaders import UnstructuredHTMLLoader

def load_html_basic(file_path: str) -> List:

    print(f"Loading {file_path}...", end="")
    loader = UnstructuredHTMLLoader(file_path)
    docs = loader.load()
    print("Done")
    return docs
from typing import List
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

def load_docx_basic(file_path: str) -> List:
    """
    Loads a Microsoft Word document using Langchain's Unstructured document loader.

    Args:
        file_path: The path of the file on the local machine.

    Returns:
        a list of LangChain's Document objects
    """
    print(f"Loading {file_path}...", end="")
    loader = UnstructuredWordDocumentLoader(file_path)
    data = loader.load()
    print("Done")
    return data
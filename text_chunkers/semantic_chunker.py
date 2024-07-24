"""
Package pre-requisite:
langchain_experimental
langchain_openai
"""

from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def semantic_split(data: str) -> List:
    """
    The text is split by their semantic similarities in vector form, text with high similarity with its subsequent text
    will belong to the same chunk, any subsequent text with a disimilar semantic structure will belong to a different chunk.

    This is still experimental but seems to be working with some promising results.
    """
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    docs = text_splitter.create_documents([data])
    return docs
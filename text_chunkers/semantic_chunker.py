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

def documents_semantic_split(documents) -> List:
    """
    Splitting a list of documents into semantically similar chunks

    Arguments:
        documents: A list of LangChain Document objects

    Returns:
        A list of chunks
    """
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = text_splitter.split_documents(documents)

    return chunks
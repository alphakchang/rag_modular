"""
Package pre-requisite:
langchain-text-splitters
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages_recursive_character_splitter(pages: List, chunk_size: int = 512) -> List:
    """
    Use this function to chunk contents that have been loaded by a loader and have been splitted into a list of pages.

    use chunk_text_resursive_character_splitter if the content is just one string.

    This splitting technique is parameterized by a list of characters.
    It tries to split on them in order until the chunks are small enough.
    The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs
    (and then sentences, and then words) together as long as possible.

    Default chunk_size parameter is 512, but this should be changed based on the text,
    if the text tends to have longer paragraphs talking about the same topic, increase chunk_size, and vise-versa.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

def chunk_text_resursive_character_splitter(data: str, chunk_size: int = 512) -> List:
    """
    Use this function to chunk a str.

    use chunk_pages_resursive_character_splitter if the content was
    loaded by a loader and have been splitted into a list of pages.

    This splitting technique is parameterized by a list of characters.
    It tries to split on them in order until the chunks are small enough.
    The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs
    (and then sentences, and then words) together as long as possible.

    Default chunk_size parameter is 512, but this should be changed based on the text,
    if the text tends to have longer paragraphs talking about the same topic, increase chunk_size, and vise-versa.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0
    )
    chunks = text_splitter.create_documents([data])
    return chunks
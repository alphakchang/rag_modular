from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List
import chromadb
from langchain_community.vectorstores import Chroma
import chromadb.utils.embedding_functions as embedding_functions
import os

def insert_or_fetch_embeddings(collection_name: str, chunks: List[str]):
    """
    This function creates a persistent client with a local Chroma db (in the chroma_db folder)
    The collection name is then searched and either fetched or created if it does not exist.
    Finally the chunks are upserted into the collection and a ChromaDB vector store that utilises the collection provided is returned.
    """

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name="text-embedding-3-small"
            )

    chroma_client = chromadb.PersistentClient(path="../chroma_db")
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

    i = collection.count()
    chunkIds = []
    docs_content = []

    # Inputting documents into a collection requires a list of unique IDs which are generated below and provided with their respective chunks to the collection.

    for chunk in chunks:
        id = f'id_{i}'
        chunkIds.append(id)

        docs_content.append(chunk.page_content)

        i = i + 1

    collection.upsert(documents=docs_content, ids= chunkIds)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

    vectorstore = Chroma.from_texts(docs_content, embeddings, None, None, collection_name)

    return vectorstore
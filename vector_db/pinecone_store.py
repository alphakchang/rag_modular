"""
Package pre-requisite:
pinecone
langchain_pinecone
langchain_openai
"""

from typing import List
from pinecone import Pinecone, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

def insert_or_fetch_embeddings(index_name: str, chunks: List):
    """
    This function checks if an index with the given name exists,
    if it exists, the embeddings will be fetched,
    if it doesn't exist, an index with the given name will be created, and new embedding inserted.

    Returns the vector_store

    The default setting uses text-embedding-3-small model from OpenAI and produces embeddings with dimensions of 1536.
    To change the embedding model, comment in/out the embeddings variable, this has not been written into function parameter
    since we only want people who are sure about what they are doing to make any changes to the embedding model.

    Note:
    Serverless index code included but commented out, can consider writing another function specifically for this if we
    start to use serverless index in the future.
    """
    pc = Pinecone()
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072) # D3072 - Most capable embedding model for both english and non-english tasks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536) # D1536 - Increased performance over 2nd generation ada embedding model
    # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", dimensions=1536) # D1536 - Most capable 2nd generation embedding model, replacing 16 first generation models

    # check if the index exists
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ...', end='')
        vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)
        print('Done')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        
        # Pod index
        pc.create_index(
            name=index_name,
            dimension=embeddings.dimensions,
            metric='cosine',
            spec=PodSpec(environment='gcp-starter')
        )

        # Serverless index
        # from pinecone import ServerlessSpec
        # pc.create_index(
        #     name=index_name,
        #     dimension=embeddings.dimensions,
        #     metric='cosine',
        #     spec=ServerlessSpec(
        #         cloud='aws',
        #         region='us-east-1'
        #     )
        # )

        vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
        print('Done')

    return vector_store
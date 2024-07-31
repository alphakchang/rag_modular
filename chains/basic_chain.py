"""
Package pre-requisite:
langchain
langchainhub
"""

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def single_q_and_a_rag(retriever, llm):
    """
    Use this in a RAG system where only a single question & answer workflow is needed.

    For a conversational RAG workflow where conversational history is taken into account, use conversational_rag from conversational_chain.py instead.

    Parameter:
    langchain vectorstore.as_retriever() object
    langchain llm chat model object

    Returns:
    RAG chain for single Q & A
    """
    # Load the pre-defined rag prompt from langchain hub
    prompt = hub.pull("rlm/rag-prompt")

    # create a function to combine all documents into a single string, to be used in the chain
    def format_docs(docs):
        '''
        This function combines the content of all provided documents into one string,
        with each document's content separated by two newline characters.
        '''
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Use LCEL Runnable protocol to define the chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


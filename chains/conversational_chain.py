"""
Package pre-requisite:
langchain
langchainhub
"""

from langchain import hub
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from chains import basic_chain


def conversational_rag(retriever, llm):
    """
    Use this in a conversational RAG workflow where conversational history is taken into account.

    Parameter:
    langchain vectorstore.as_retriever() object

    Returns:
    RAG chain for single Q & A
    """

    # Load the pre-defined prompt for contextualization from langchain hub
    prompt = hub.pull("albint3r/contextualize-prompt")

    # create the contextualize_prompt object
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt.messages[0].prompt.template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # create the history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    rag_chain = basic_chain.single_q_and_a_rag(history_aware_retriever, llm)

    return rag_chain
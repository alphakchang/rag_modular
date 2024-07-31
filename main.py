from dotenv import load_dotenv, find_dotenv
from content_loaders import pdf_loader
from text_chunkers import character_chunker, semantic_chunker
from vector_db import pinecone_store
from langchain_openai import ChatOpenAI
from chains import conversational_chain
from langchain_core.messages import AIMessage, HumanMessage


load_dotenv(find_dotenv())

def create_text_rag(file_path: str):
    """
    Use this function to create a RAG from a text file.
    """
    # 1. Load the text file
    with open(file_path, encoding="utf-8") as f:
        text_file = f.read()
    
    # 2. Split the text semantically
    chunks = semantic_chunker.semantic_split(text_file)
    print(f"Successfully split into {len(chunks)} chunks.")

    # 3. Get the vector store ready
    index_name = 'memoq-test'
    vector_store = pinecone_store.insert_or_fetch_embeddings(index_name, chunks)
    
    # 4. create the retriever, setting k=3, adjust as appropriate
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # 5. define the llm
    llm = ChatOpenAI(model="gpt-4o")

    # 6. create conversational rag chain
    rag_chain = conversational_chain.conversational_rag(retriever, llm)
    

    return rag_chain

if __name__ == "__main__":
    file_path = "data/memoQ_troubleshoot.txt"
    rag_chain = create_text_rag(file_path)

    # for chunk in rag_chain.stream("My memoQ stopped working"):
    #     print(chunk, end="", flush=True)

    ### Testing conversational rag with two questions ###

    chat_history = []

    # question 1
    question_1 = "my memoQ stopped working"
    print(f"Human Question: {question_1}\n")

    ai_msg_1 = rag_chain.invoke({"input": question_1, "chat_history": chat_history})
    print(f"AI reponse: {ai_msg_1}\n\n")

    # add AI response from question 1 into the history
    chat_history.extend(
        [
            HumanMessage(content=question_1),
            AIMessage(content=ai_msg_1),
        ]
    )

    # question 2
    question_2 = "are you sure?"
    print(f"Human Question: {question_2}\n")

    ai_msg_2 = rag_chain.invoke({"input": question_2, "chat_history": chat_history})

    print(f"AI reponse: {ai_msg_2}\n\n")

    # add AI response from question 2 into the history
    chat_history.extend(
        [
            HumanMessage(content=question_2),
            AIMessage(content=ai_msg_2),
        ]
    )

    # question 3
    question_3 = "I think you are wrong!"
    print(f"Human Question: {question_3}\n")

    ai_msg_3 = rag_chain.invoke({"input": question_3, "chat_history": chat_history})

    print(f"AI reponse: {ai_msg_3}\n\n")

    # add AI response from question 3 into the history
    chat_history.extend(
        [
            HumanMessage(content=question_3),
            AIMessage(content=ai_msg_3),
        ]
    )

    print(chat_history)
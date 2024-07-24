from dotenv import load_dotenv, find_dotenv
from content_loaders import pdf_loader
from text_chunkers import character_chunker, semantic_chunker
from vector_db import pinecone_store
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

    # 4. Load the pre-defined rag prompt from langchain hub
    prompt = hub.pull("rlm/rag-prompt")
    
    # 5. create the retriever, setting k=3, adjust as appropriate
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 6. create the LLM
    llm = ChatOpenAI(model="gpt-4o")

    # 7. create a function to combine all documents into a single string, to be used in the chain
    def format_docs(docs):
        '''
        This function combines the content of all provided documents into one string,
        with each document's content separated by two newline characters.
        '''
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 8. Use LCEL Runnable protocol to define the chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    file_path = "data/memoQ_troubleshoot.txt"
    rag_chain = create_text_rag(file_path)

    for chunk in rag_chain.stream("My memoQ stopped working"):
        print(chunk, end="", flush=True)
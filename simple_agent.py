import os
import gradio as gr
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to answer questions
    about a provided PDF document, powered by LangChain.
    """
    def __init__(self, pdf_path: str):
        """
        Initializes the agent by setting up the LangChain retrieval and generation chain.
        """
        try:
            # This check is still good practice, as LangChain will need it.
            if "GOOGLE_API_KEY" not in os.environ:
                raise KeyError
        except KeyError:
            raise RuntimeError(
                "Error: GOOGLE_API_KEY environment variable not set. "
                "Please get your key from Google AI Studio and set the environment variable."
            )

        print("Initializing LangChain-based agent...")

        # 1. Set up the LLM and Embeddings model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # 2. Load and process the document
        print(f"Loading and processing {pdf_path}...")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        # 3. Create the vector store and retriever
        print("Creating vector store from document chunks...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        # 4. Create the generation chain
        system_prompt = """
You are a helpful assistant. Answer the following question based ONLY on the provided context from a document. If the context doesn't contain the answer, say "I could not find the answer in the document."

<context>
{context}
</context>

Question: {input}
"""
        prompt = ChatPromptTemplate.from_template(system_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        print(" Agent is ready. You can start asking questions now.")

    def get_response(self, user_input: str, history: list) -> str:
        """
        Invokes the LangChain RAG chain to get a response.
        Note: History is not explicitly used here as the RAG chain is stateless per query,
        but it's kept for potential future stateful implementations.
        """
        print("🤖 Thinking...")
        response = self.rag_chain.invoke({"input": user_input})
        return response["answer"]

def main():
    """Main function to create and run the agent."""
    # IMPORTANT: Replace this with the actual path to your PDF file.
    # Using a raw string (r"...") is recommended on Windows to handle backslashes.
    pdf_file_path = r"C:\Users\EliteBook\Downloads\AI Data Science Trainer Test (2).pdf"

    if not os.path.exists(pdf_file_path):
        print(f"Error: The file was not found at the path: {pdf_file_path}")
        print("Please make sure the file path is correct and try again.")
        return

    try:
        agent = RAGAgent(pdf_path=pdf_file_path)

        # This wrapper function is what Gradio will call.
        def chat_interface_fn(message, history):
            return agent.get_response(message, history)

        iface = gr.ChatInterface(
            fn=chat_interface_fn,
            title="📄 Document Q&A Agent",
            description="Ask questions about your PDF document. The agent will use the document content to answer."
        )
        iface.launch()

    except RuntimeError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
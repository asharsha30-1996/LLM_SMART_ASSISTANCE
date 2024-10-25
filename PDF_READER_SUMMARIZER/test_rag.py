import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil

# Load environment variables and configure the API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please check your .env file.")
    st.stop()

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files with detailed logging and feedback."""
    all_text = ""  # Store all text from all PDFs
    total_docs = len(pdf_docs)  # Total PDFs to process
    progress_bar = st.progress(0)  # Initialize progress bar

    for idx, pdf in enumerate(pdf_docs):
        pdf_text = ""  # Store text for the current PDF
        try:
            st.info(f"Processing {pdf.name} ({idx + 1}/{total_docs})...")
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {e}. Skipping this file.")
            continue  # Skip to the next file in case of an error

        if pdf_text.strip():  # Only add if the PDF has valid content
            all_text += pdf_text + "\n\n"
            st.success(f"{pdf.name} processed successfully.")
        else:
            st.warning(f"No content found in {pdf.name}. Skipping.")

        # Update the progress bar for each processed PDF
        progress_bar.progress((idx + 1) / total_docs)

    if not all_text.strip():
        st.warning("No text extracted from the uploaded PDFs.")
    return all_text

def get_text_chunks(text):
    """Split text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def create_faiss_index(text_chunks):
    """Create and save FAISS vector store from text chunks."""
    try:
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")  # Remove old index directory
    except PermissionError as e:
        st.error(f"PermissionError: {e}. Close any program accessing 'faiss_index' and try again.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create a conversational chain using OpenAI model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, respond with 
    "answer is not available in the context." Do not provide incorrect answers.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Process user questions and generate a response."""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            st.warning("No relevant documents found.")
            return

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response.get("output_text", "No response generated."))
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with PDF using OpenAI", page_icon="💬", layout="wide")
    st.header("Chat with PDF using OpenAI 💁")

    if 'processed' not in st.session_state:
        st.session_state.processed = False

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and st.session_state.processed:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)  # Step 1: Extract all PDF text
                if raw_text.strip():  # Ensure text was extracted
                    text_chunks = get_text_chunks(raw_text)  # Step 2: Split text into chunks
                    create_faiss_index(text_chunks)  # Step 3: Create and save FAISS index
                    st.session_state.processed = True
                    st.success("Processing complete! You can now ask questions.")

    if st.session_state.processed:
        st.info("PDFs are processed. You can now ask questions.")

if __name__ == "__main__":
    main()

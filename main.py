import os
import fitz
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone

import pinecone

# --- Load environment variables ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "medical-chatbot"

# --- Step 1 & 2: PDF Extraction ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        print(f"✅ Extracted text from page {page_num}")
        full_text += text + "\n"
    doc.close()
    return full_text

# --- Step 3: Chunking ---
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    print(f"\n✅ Total Chunks Created: {len(chunks)}")
    return chunks

# --- Step 4: Embedding and storing in Pinecone ---
def store_embeddings_in_pinecone(chunks):
    # Initialize Hugging Face embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    # Check if index exists, create if not
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=INDEX_NAME, dimension=384, metric="cosine")

    # Create vectorstore from chunks and upload
    docsearch = Pinecone.from_documents(documents=chunks, embedding=embeddings, index_name=INDEX_NAME)

    print("✅ All chunks embedded and uploaded to Pinecone.")


if __name__ == "__main__":
    pdf_path = "medical_book.pdf"
    text = extract_text_from_pdf(pdf_path)
    
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("✅ Text saved to extracted_text.txt")

    chunks = split_text_into_chunks(text)

    # Optional preview
    with open("chunks_preview.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks[:5]):
            f.write(f"--- Chunk {i+1} ---\n{chunk.page_content}\n\n")

    store_embeddings_in_pinecone(chunks)

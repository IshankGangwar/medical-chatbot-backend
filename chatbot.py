import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "medical-chatbot"

# Initialize FastAPI
app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Create Retriever
def create_retriever():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# 🤖 LLM (Groq)
def create_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

# 🔁 Retrieval-QA Chain
def create_chatbot():
    retriever = create_retriever()
    llm = create_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a helpful medical assistant. Use the context below to answer the user's question.
Do not say "according to textbook" — just answer clearly and concisely.

If the answer is not in the context, say: "Sorry, I don’t have knowledge about that."

Keep your response under 50–60 words.

Context:
{context}

Question: {question}
""")
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

qa_chain = create_chatbot()

# 📥 Request Body Schema
class QueryInput(BaseModel):
    question: str

# 📤 POST endpoint to get answer
@app.post("/query")
async def query(input: QueryInput):
    try:
        question = input.question
        print(f"🔍 Question received: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"✅ Answer: {result['result']}")
        return {"answer": result["result"]}
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {"answer": f"Internal server error: {str(e)}"}

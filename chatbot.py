import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
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

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For now allow all, secure later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup retriever
def create_retriever():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Setup LLM and chain
def create_chatbot():
    retriever = create_retriever()
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_template("""
You are a helpful medical assistant. Use the context below to answer the user's question.
If the answer is not in the context, say: "Sorry, I don’t have knowledge about that."
Keep your response under 50–60 words.
You are a medical assistant. Answer only medical-related queries. If the user asks anything unrelated (like quotes or jokes), politely say you only respond to medical questions.\n\nUser: {question}\nBot

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

@app.post("/query")
async def query(request: Request):
    try:
        body = await request.json()
        question = body.get("question", "")
        if not question:
            return {"answer": "Please ask a question."}
        result = qa_chain.invoke({"query": question})
        return {"answer": result["result"]}
    except Exception as e:
        return {"answer": f"Internal error: {str(e)}"}

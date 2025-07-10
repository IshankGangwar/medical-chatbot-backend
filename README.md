# 🧠 MediBot Backend – LLM-Powered Medical Chat API

This is the backend API for MediBot – an AI-powered chatbot that answers **medical-related queries** using a pre-trained LLM hosted on Hugging Face. Built with **FastAPI**, this server processes user questions and returns AI-generated medical responses.

---

## 🚀 Live API Endpoint (Deployed on Hugging Face Spaces)

https://huggingface.co/spaces/ishank22/medibot-backend


---

## ⚙️ Features

- ✅ LLM-based answer generation (uses sentence-transformers model)
- ✅ Lightweight FastAPI server with single `/query` POST endpoint
- ✅ Handles both medical questions and custom fallback replies
- ✅ Ready for integration with any frontend via JSON API

---

## 🧱 Tech Stack

- FastAPI + Uvicorn
- LangChain + HuggingFaceEmbeddings
- Docker (for Hugging Face deployment)
- Model: `sentence-transformers/all-MiniLM-L6-v2`

---

## 📂 Project Structure

medibot-backend/
├── chatbot.py # Main FastAPI app with LangChain logic
├── requirements.txt # Python dependencies
├── Dockerfile # Containerization for Hugging Face Spaces
├── models/ # Pre-downloaded embedding model
├── README.md


---

## 🧪 Test Locally

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run server locally
uvicorn chatbot:app --reload

Access local API at:
http://localhost:8000/query

Deployment (on Hugging Face Spaces)
Upload all files to your HF Space with Docker template.

Ensure the models/all-MiniLM-L6-v2/ folder is included.

Space will auto-run using Dockerfile.

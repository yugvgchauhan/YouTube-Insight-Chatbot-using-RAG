"""
Configuration management for YouTube RAG Chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# HuggingFace Configuration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
HUGGINGFACE_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HUGGINGFACE_EMBEDDING_MODEL = "intfloat/e5-small-v2"

# OpenAI Configuration (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
CHROMA_COLLECTION_NAME = "youtube_transcripts"

# Text Splitting Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# RAG Configuration
RETRIEVAL_K = 4  # Number of documents to retrieve
LLM_TEMPERATURE = 0.2
LLM_MAX_NEW_TOKENS = 512

# Streamlit Configuration
PAGE_TITLE = "YouTube Video Chatbot"
PAGE_ICON = "🎥"


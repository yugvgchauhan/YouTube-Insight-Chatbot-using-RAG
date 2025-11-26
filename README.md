# YouTube Video RAG Chatbot

A Streamlit-based chatbot application that uses RAG (Retrieval-Augmented Generation) to answer questions about YouTube videos. The app fetches video transcripts, creates embeddings using HuggingFace models, stores them in ChromaDB, and uses a HuggingFace LLM to generate answers.

## Features

- 🎥 Extract video ID from YouTube URLs automatically
- 📝 Fetch and process YouTube video transcripts
- 🔍 Create vector embeddings using HuggingFace models
- 💾 Store embeddings in ChromaDB for efficient retrieval
- 🤖 Answer questions using RAG with HuggingFace LLM
- 💬 Interactive chat interface with Streamlit
- ⚡ Caching of vector stores for faster subsequent queries

## Project Structure

```
youtube_analysis/
├── .env                          # Environment variables (API tokens)
├── .env.example                  # Example .env file template
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── app.py                        # Main Streamlit application
├── config.py                     # Configuration management
├── src/
│   ├── __init__.py
│   ├── transcript_fetcher.py    # YouTube transcript fetching logic
│   ├── vector_store.py           # ChromaDB vector store management
│   ├── rag_chain.py             # RAG chain creation and querying
│   └── utils.py                 # Utility functions (URL parsing, etc.)
└── chroma_db/                    # ChromaDB storage directory (auto-created)
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

1. Copy the example environment file:
   ```bash
   copy .env.example .env  # Windows
   # or
   cp .env.example .env   # macOS/Linux
   ```

2. Edit `.env` and add your HuggingFace API token:
   ```
   HUGGINGFACE_API_TOKEN=your_actual_token_here
   ```

   To get a HuggingFace API token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with read permissions
   - Copy and paste it into your `.env` file

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Enter YouTube Video**: 
   - Go to the "Video Input" tab
   - Paste a YouTube URL (e.g., `https://www.youtube.com/watch?v=VIDEO_ID`) or enter a video ID
   - Click "Fetch Transcript"

2. **Chat with Video**:
   - Switch to the "Chat" tab
   - Ask questions about the video content
   - The chatbot will answer based on the video transcript

## Supported YouTube URL Formats

The app automatically extracts video IDs from various URL formats:
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID`
- Direct video ID: `VIDEO_ID`

## Configuration

You can modify settings in `config.py`:

- **Embedding Model**: Change `HUGGINGFACE_EMBEDDING_MODEL`
- **LLM Model**: Change `HUGGINGFACE_LLM_MODEL`
- **Chunk Size**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP`
- **Retrieval**: Modify `RETRIEVAL_K` (number of documents to retrieve)
- **LLM Parameters**: Adjust `LLM_TEMPERATURE` and `LLM_MAX_NEW_TOKENS`

## Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: RAG pipeline and chain management
- **ChromaDB**: Vector database for embeddings
- **HuggingFace**: Embeddings and LLM models
- **YouTube Transcript API**: Fetching video transcripts

## Notes

- The vector store is cached in the `chroma_db/` directory for faster subsequent queries
- Transcripts are fetched in English by default, but the app tries to find the best available transcript
- The app requires an active internet connection to fetch transcripts and use HuggingFace models

## Troubleshooting

1. **"HUGGINGFACE_API_TOKEN not found"**: Make sure your `.env` file exists and contains a valid token
2. **"No transcript found"**: The video may not have transcripts available
3. **"Transcripts are disabled"**: The video owner has disabled transcripts
4. **Slow responses**: The first query may be slow as the LLM loads. Subsequent queries should be faster

## License

This project is for educational purposes.


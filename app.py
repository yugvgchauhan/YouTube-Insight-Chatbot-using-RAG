"""
Streamlit App for YouTube Video RAG Chatbot
"""
import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.transcript_fetcher import TranscriptFetcher
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGChain
from src.utils import extract_video_id
import config

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide"
)

# Initialize session state
if 'transcript_fetched' not in st.session_state:
    st.session_state.transcript_fetched = False
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🎥 YouTube Video Chatbot")
    st.markdown("Ask questions about any YouTube video using AI-powered RAG (Retrieval-Augmented Generation)")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check for API token
        if not config.HUGGINGFACE_API_TOKEN:
            st.error("⚠️ HUGGINGFACE_API_TOKEN not found!")
            st.info("Please set your HuggingFace API token in the .env file")
            st.stop()
        else:
            st.success("✅ HuggingFace API token loaded")
        
        st.markdown("---")
        st.markdown("### 📝 Settings")
        st.info(f"**Embedding Model:** {config.HUGGINGFACE_EMBEDDING_MODEL}")
        st.info(f"**LLM Model:** {config.HUGGINGFACE_LLM_MODEL}")
        st.info(f"**Chunk Size:** {config.CHUNK_SIZE}")
        st.info(f"**Retrieval K:** {config.RETRIEVAL_K}")
    
    # Main content area
    tab1, tab2 = st.tabs(["📹 Video Input", "💬 Chat"])
    
    with tab1:
        st.header("Enter YouTube Video")
        
        # Input for YouTube URL or ID
        url_input = st.text_input(
            "YouTube URL or Video ID",
            placeholder="https://www.youtube.com/watch?v=VIDEO_ID or VIDEO_ID",
            help="Paste a YouTube URL or enter a video ID"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            fetch_button = st.button("🔍 Fetch Transcript", type="primary", use_container_width=True)
        
        with col2:
            if st.session_state.video_id:
                clear_button = st.button("🗑️ Clear", use_container_width=True)
                if clear_button:
                    # Clear session state
                    st.session_state.transcript_fetched = False
                    st.session_state.vector_store_ready = False
                    st.session_state.rag_chain = None
                    st.session_state.video_id = None
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Process video input
        if fetch_button and url_input:
            with st.spinner("Processing video..."):
                try:
                    # Extract video ID
                    video_id = extract_video_id(url_input)
                    
                    if not video_id:
                        st.error("❌ Invalid YouTube URL or video ID. Please check and try again.")
                    else:
                        st.session_state.video_id = video_id
                        
                        # Display video info
                        st.success(f"✅ Video ID: `{video_id}`")
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        st.markdown(f"**Video Link:** [Watch on YouTube]({video_url})")
                        
                        # Fetch transcript
                        fetcher = TranscriptFetcher()
                        transcript = fetcher.fetch_transcript(video_id)
                        
                        if transcript:
                            st.session_state.transcript_fetched = True
                            
                            # Display transcript stats
                            word_count = len(transcript.split())
                            st.info(f"📊 Transcript loaded: {word_count:,} words")
                            
                            # Create or load vector store
                            with st.spinner("Creating vector store..."):
                                vector_manager = VectorStoreManager(video_id)
                                
                                # Try to load existing vector store
                                vector_store = vector_manager.load_vector_store()
                                
                                if vector_store is None:
                                    # Create new vector store
                                    vector_store = vector_manager.create_vector_store(transcript)
                                    st.success("✅ Vector store created successfully!")
                                else:
                                    st.success("✅ Vector store loaded from cache!")
                                
                                # Create retriever
                                retriever = vector_manager.get_retriever()
                                
                                # Initialize RAG chain
                                with st.spinner("Initializing RAG chain..."):
                                    st.session_state.rag_chain = RAGChain(retriever)
                                    st.session_state.vector_store_ready = True
                                    st.success("✅ Ready to chat! Switch to the Chat tab.")
                        
                except ValueError as e:
                    st.error(f"❌ Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.exception(e)
        
        # Display current status
        if st.session_state.video_id:
            st.markdown("---")
            st.markdown("### 📊 Current Status")
            status_col1, status_col2, status_col3 = st.columns(3)
            
            with status_col1:
                if st.session_state.transcript_fetched:
                    st.success("✅ Transcript")
                else:
                    st.warning("⏳ Transcript")
            
            with status_col2:
                if st.session_state.vector_store_ready:
                    st.success("✅ Vector Store")
                else:
                    st.warning("⏳ Vector Store")
            
            with status_col3:
                if st.session_state.rag_chain:
                    st.success("✅ RAG Chain")
                else:
                    st.warning("⏳ RAG Chain")
    
    with tab2:
        st.header("Chat with Video")
        
        if not st.session_state.vector_store_ready or st.session_state.rag_chain is None:
            st.warning("⚠️ Please fetch a video transcript first in the 'Video Input' tab.")
        else:
            # Display video info
            if st.session_state.video_id:
                st.info(f"📹 Chatting about video: `{st.session_state.video_id}`")
            
            # Display chat history
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)
            
            # Chat input
            user_question = st.chat_input("Ask a question about the video...")
            
            if user_question:
                # Add user question to chat
                st.session_state.chat_history.append((user_question, None))
                
                # Get answer from RAG chain
                with st.spinner("Thinking..."):
                    try:
                        answer = st.session_state.rag_chain.query(user_question)
                        # Update the last entry with the answer
                        st.session_state.chat_history[-1] = (user_question, answer)
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.session_state.chat_history[-1] = (user_question, error_msg)
                        st.rerun()


if __name__ == "__main__":
    main()


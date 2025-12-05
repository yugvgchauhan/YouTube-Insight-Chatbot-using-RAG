"""
Streamlit App for YouTube Video RAG Chatbot
"""
import ast
import json
import os
import re
import sys

import streamlit as st

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
if "transcript_fetched" not in st.session_state:
    st.session_state.transcript_fetched = False
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "chat_history" not in st.session_state:
    # Each entry: {"question": str, "answer": str, "sources": List[Document]}
    st.session_state.chat_history = []
if "transcript_language" not in st.session_state:
    st.session_state.transcript_language = "en"
if "answer_language" not in st.session_state:
    st.session_state.answer_language = "en"
if "retrieval_strategy" not in st.session_state:
    st.session_state.retrieval_strategy = "similarity"
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = config.RETRIEVAL_K
if "use_multi_query" not in st.session_state:
    st.session_state.use_multi_query = False
if "answer_tone" not in st.session_state:
    st.session_state.answer_tone = "neutral"
if "answer_style" not in st.session_state:
    st.session_state.answer_style = "auto"
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = None


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🎥 YouTube Video Chatbot")
    st.markdown(
        "Ask questions about any YouTube video using AI-powered RAG (Retrieval-Augmented Generation)"
    )
    
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

        st.markdown("---")
        st.markdown("### 🌐 Language")
        st.session_state.transcript_language = st.selectbox(
            "Transcript language (preferred)",
            options=["auto", "en", "hi", "es", "fr", "de"],
            index=1,
            help="Preferred language to fetch the transcript in. 'auto' lets YouTube choose the best available language.",
        )
        st.session_state.answer_language = st.selectbox(
            "Answer language",
            options=["en", "hi", "es", "fr", "de"],
            index=0,
            help="Language for the chatbot answers.",
        )

        st.markdown("---")
        st.markdown("### 🔍 Retrieval")
        st.session_state.retrieval_strategy = st.selectbox(
            "Retrieval strategy",
            options=["similarity", "mmr"],
            index=0,
            help="Use 'mmr' (Maximum Marginal Relevance) to increase diversity of retrieved chunks.",
        )
        st.session_state.retrieval_k = st.slider(
            "Number of chunks to retrieve (k)",
            min_value=1,
            max_value=10,
            value=config.RETRIEVAL_K,
            help="Controls how many transcript chunks are retrieved for each question.",
        )
        st.session_state.use_multi_query = st.checkbox(
            "Use multi-query expansion",
            value=False,
            help="Generate multiple related queries to improve retrieval coverage.",
        )

        st.markdown("---")
        st.markdown("### ✍️ Answer Style")
        st.session_state.answer_tone = st.selectbox(
            "Tone",
            options=["neutral", "teacher", "friendly", "exam coach"],
            index=0,
        )
        st.session_state.answer_style = st.selectbox(
            "Style",
            options=["auto", "concise", "detailed", "bullet points", "step-by-step"],
            index=0,
        )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📹 Video Input", "💬 Chat", "🧪 Quiz"])
    
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
                    st.session_state.transcript_text = None
                    st.session_state.quiz_questions = []
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None
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

                        # Prepare language preferences
                        transcript_language = st.session_state.transcript_language
                        answer_language = st.session_state.answer_language
                        retrieval_strategy = st.session_state.retrieval_strategy
                        retrieval_k = st.session_state.retrieval_k
                        use_multi_query = st.session_state.use_multi_query
                        answer_tone = st.session_state.answer_tone
                        answer_style = st.session_state.answer_style

                        preferred_languages = (
                            None
                            if transcript_language == "auto"
                            else [transcript_language]
                        )
                        
                        # Fetch transcript
                        fetcher = TranscriptFetcher()
                        transcript = fetcher.fetch_transcript(
                            video_id,
                            language_codes=preferred_languages,
                            target_language=answer_language,
                        )
                        
                        if transcript:
                            st.session_state.transcript_fetched = True
                            st.session_state.transcript_text = transcript
                            
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
                                search_kwargs = None
                                if retrieval_strategy == "mmr":
                                    search_kwargs = {
                                        "k": retrieval_k,
                                        "fetch_k": max(retrieval_k * 4, retrieval_k),
                                    }

                                retriever = vector_manager.get_retriever(
                                    k=retrieval_k,
                                    search_type=retrieval_strategy,
                                    search_kwargs=search_kwargs,
                                )
                                
                                # Initialize RAG chain
                                with st.spinner("Initializing RAG chain..."):
                                    st.session_state.rag_chain = RAGChain(
                                        retriever=retriever,
                                        answer_language=answer_language,
                                        answer_tone=answer_tone,
                                        answer_style=answer_style,
                                        use_multi_query=use_multi_query,
                                    )
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
            for i, message in enumerate(st.session_state.chat_history):
                question = message.get("question")
                answer = message.get("answer")
                sources = message.get("sources", [])

                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)
                    if sources:
                        with st.expander("📚 Sources used for this answer"):
                            for idx, doc in enumerate(sources, start=1):
                                metadata = getattr(doc, "metadata", {}) or {}
                                chunk_index = metadata.get("chunk_index")
                                st.markdown(f"**Source {idx}**")
                                st.write(doc.page_content)
                                if chunk_index is not None:
                                    st.caption(f"Chunk index: {chunk_index}")
                                st.markdown("---")
            
            # Chat input
            user_question = st.chat_input("Ask a question about the video...")
            
            if user_question:
                # Add user question to chat
                st.session_state.chat_history.append(
                    {"question": user_question, "answer": None, "sources": []}
                )
                
                # Get answer from RAG chain (with sources for grounding)
                with st.spinner("Thinking..."):
                    try:
                        answer, sources = st.session_state.rag_chain.query_with_sources(
                            user_question
                        )
                        # Update the last entry with the answer and sources
                        st.session_state.chat_history[-1] = {
                            "question": user_question,
                            "answer": answer,
                            "sources": sources,
                        }
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.session_state.chat_history[-1] = {
                            "question": user_question,
                            "answer": error_msg,
                            "sources": [],
                        }
                        st.rerun()

    with tab3:
        st.header("Quiz on Video")

        if not st.session_state.transcript_fetched or not st.session_state.transcript_text:
            st.warning("⚠️ Please fetch a video transcript first in the 'Video Input' tab.")
        else:
            st.markdown(
                "Generate a short multiple-choice quiz (10 questions) based on the video transcript."
            )

            generate_button = st.button("🧪 Generate 10-question Quiz")

            if generate_button:
                with st.spinner("Generating quiz questions..."):
                    rag_chain = st.session_state.rag_chain
                    if rag_chain is None:
                        st.error("RAG chain not initialized. Please refetch the video.")
                    else:
                        # Get diverse transcript chunks using the retriever
                        # Use multiple queries to get diverse coverage
                        diverse_queries = [
                            "main topics and concepts",
                            "key points explained",
                            "important details",
                            "examples and demonstrations",
                            "conclusions and summaries"
                        ]
                        
                        all_chunks = []
                        seen_content = set()
                        for query in diverse_queries:
                            try:
                                docs = rag_chain._call_retriever(query)
                                if not isinstance(docs, list):
                                    docs = [docs]
                                for doc in docs:
                                    content = getattr(doc, "page_content", "")
                                    if content and content not in seen_content:
                                        seen_content.add(content)
                                        all_chunks.append(content)
                            except Exception:
                                continue
                        
                        # Limit total context length (use first ~3000 words)
                        combined_context = "\n\n".join(all_chunks[:20])  # Limit to 20 chunks
                        words = combined_context.split()
                        if len(words) > 3000:
                            combined_context = " ".join(words[:3000])
                        
                        # Format prompt for chat model - emphasize 10 questions multiple times
                        prompt_text = (
                            "You are a helpful exam generator for a YouTube tutorial.\n"
                            "CRITICAL: You MUST generate EXACTLY 10 multiple-choice questions. Not 3, not 5, not 7 - EXACTLY 10 questions.\n"
                            "Given the following transcript excerpts from a video, generate EXACTLY 10 "
                            "multiple-choice questions that test understanding of the key concepts.\n\n"
                            "Requirements:\n"
                            "- Generate EXACTLY 10 questions (this is mandatory).\n"
                            "- Cover different important topics from the transcript.\n"
                            "- Each question must have exactly 4 answer options.\n"
                            "- Only one option is correct.\n"
                            "- Questions should test understanding, not trivial facts.\n"
                            "- Make sure correct_index is 0, 1, 2, or 3 for each question.\n\n"
                            "Return ONLY a valid JSON array with EXACTLY 10 question objects, no other text:\n"
                            "[\n"
                            "  {\n"
                            '    "question": "...",\n'
                            '    "options": ["option A", "option B", "option C", "option D"],\n'
                            '    "correct_index": 0,\n'
                            '    "explanation": "Short explanation."\n'
                            "  },\n"
                            "  ... (repeat for all 10 questions)\n"
                            "]\n\n"
                            "Remember: The array must contain EXACTLY 10 question objects.\n\n"
                            "Transcript excerpts:\n"
                            f"{combined_context}\n"
                        )
                        
                        try:
                            # Use the chat model properly
                            from langchain_core.messages import HumanMessage
                            
                            message = HumanMessage(content=prompt_text)
                            raw = rag_chain.llm.invoke([message])
                            content = getattr(raw, "content", str(raw))
                            
                            # Helper function to clean and fix JSON
                            def clean_json_string(text):
                                """Try to fix common JSON issues."""
                                # Remove markdown code blocks if present
                                text = re.sub(r'```json\s*', '', text)
                                text = re.sub(r'```\s*', '', text)
                                
                                # Find JSON array boundaries
                                start = text.find("[")
                                end = text.rfind("]")
                                if start != -1 and end != -1 and end > start:
                                    text = text[start:end+1]
                                
                                # Fix common issues: unescaped quotes in strings
                                # This is a simple fix - replace single quotes with escaped double quotes in string values
                                # But be careful not to break the structure
                                
                                return text.strip()
                            
                            # Try multiple parsing strategies
                            questions = None
                            json_str = clean_json_string(content)
                            
                            # Strategy 1: Try strict JSON parsing
                            try:
                                questions = json.loads(json_str)
                            except json.JSONDecodeError as e:
                                # Strategy 2: Try Python literal eval (handles single quotes)
                                try:
                                    questions = ast.literal_eval(json_str)
                                except (ValueError, SyntaxError):
                                    # Strategy 3: Try to extract individual question objects using regex
                                    # Find all question-like objects
                                    question_pattern = r'\{[^{}]*"question"[^{}]*\}'
                                    matches = re.findall(question_pattern, json_str, re.DOTALL)
                                    if matches:
                                        questions = []
                                        for match in matches:
                                            try:
                                                q = json.loads(match)
                                                questions.append(q)
                                            except:
                                                try:
                                                    q = ast.literal_eval(match)
                                                    questions.append(q)
                                                except:
                                                    continue
                                    
                                    # If still no questions, show error with raw content for debugging
                                    if not questions:
                                        st.error(f"Failed to parse JSON. Raw output preview:\n{content[:500]}...")
                                        raise ValueError(f"Could not parse quiz questions from LLM output: {str(e)}")

                            valid_questions = []
                            for q in questions:
                                if (
                                    isinstance(q, dict)
                                    and "question" in q
                                    and "options" in q
                                    and "correct_index" in q
                                    and isinstance(q["options"], list)
                                    and len(q["options"]) >= 2
                                ):
                                    valid_questions.append(q)

                            if not valid_questions:
                                st.error("Failed to parse quiz questions from model output.")
                            else:
                                # If we got fewer than 10 questions, try generating more
                                if len(valid_questions) < 10:
                                    st.warning(f"⚠️ Only {len(valid_questions)} questions were generated. Attempting to generate more...")
                                    
                                    # Try to generate additional questions
                                    remaining_needed = 10 - len(valid_questions)
                                    additional_prompt = (
                                        f"You already generated {len(valid_questions)} questions. "
                                        f"Now generate EXACTLY {remaining_needed} more multiple-choice questions "
                                        f"based on the same transcript. Return ONLY a JSON array with {remaining_needed} question objects:\n\n"
                                        f"Transcript excerpts:\n{combined_context}\n"
                                    )
                                    
                                    try:
                                        additional_message = HumanMessage(content=additional_prompt)
                                        additional_raw = rag_chain.llm.invoke([additional_message])
                                        additional_content = getattr(additional_raw, "content", str(additional_raw))
                                        additional_json_str = clean_json_string(additional_content)
                                        
                                        try:
                                            additional_questions = json.loads(additional_json_str)
                                        except json.JSONDecodeError:
                                            try:
                                                additional_questions = ast.literal_eval(additional_json_str)
                                            except:
                                                additional_questions = []
                                        
                                        # Validate and add additional questions
                                        for q in additional_questions:
                                            if (
                                                isinstance(q, dict)
                                                and "question" in q
                                                and "options" in q
                                                and "correct_index" in q
                                                and isinstance(q["options"], list)
                                                and len(q["options"]) >= 2
                                                and len(valid_questions) < 10
                                            ):
                                                valid_questions.append(q)
                                    except Exception as e:
                                        st.warning(f"Could not generate additional questions: {str(e)}")
                                
                                # Store questions (up to 10)
                                st.session_state.quiz_questions = valid_questions[:10]
                                st.session_state.quiz_submitted = False
                                st.session_state.quiz_score = None
                                
                                if len(st.session_state.quiz_questions) == 10:
                                    st.success(
                                        f"✅ Generated {len(st.session_state.quiz_questions)} questions. Scroll down to take the quiz."
                                    )
                                else:
                                    st.warning(
                                        f"⚠️ Generated {len(st.session_state.quiz_questions)} questions (requested 10). "
                                        "You can still take the quiz with these questions."
                                    )
                        except Exception as e:
                            st.error(f"Error generating quiz: {str(e)}")

            if st.session_state.quiz_questions:
                st.markdown("---")
                st.subheader("Quiz")

                for idx, q in enumerate(st.session_state.quiz_questions):
                    st.markdown(f"**Q{idx + 1}. {q['question']}**")
                    options = q.get("options", [])
                    if not options:
                        continue

                    option_indices = list(range(len(options)))

                    def _format_opt(i, opts=options):
                        label = chr(65 + i) if i < 26 else str(i + 1)
                        return f"{label}. {opts[i]}"

                    st.radio(
                        "Select an answer:",
                        options=option_indices,
                        format_func=_format_opt,
                        key=f"quiz_q_{idx}",
                    )
                    st.markdown("")  # spacing

                submit_quiz = st.button("✅ Submit Quiz")

                if submit_quiz:
                    total = len(st.session_state.quiz_questions)
                    correct = 0
                    details = []

                    for idx, q in enumerate(st.session_state.quiz_questions):
                        options = q.get("options", [])
                        correct_index = q.get("correct_index", 0)
                        explanation = q.get("explanation", "")
                        selected = st.session_state.get(f"quiz_q_{idx}", None)

                        is_correct = selected == correct_index
                        if is_correct:
                            correct += 1

                        details.append(
                            {
                                "question": q.get("question", ""),
                                "correct": is_correct,
                                "correct_option": options[correct_index]
                                if 0 <= correct_index < len(options)
                                else None,
                                "selected_option": options[selected]
                                if isinstance(selected, int)
                                and 0 <= selected < len(options)
                                else None,
                                "explanation": explanation,
                            }
                        )

                    st.session_state.quiz_submitted = True
                    st.session_state.quiz_score = {
                        "total": total,
                        "correct": correct,
                        "details": details,
                    }

                if st.session_state.quiz_submitted and st.session_state.quiz_score:
                    score = st.session_state.quiz_score
                    total = score["total"]
                    correct = score["correct"]
                    st.markdown("---")
                    st.subheader("Results")
                    st.success(
                        f"You scored {correct} out of {total} ({correct * 100 / total:.1f}%)."
                    )

                    with st.expander("View detailed feedback"):
                        for idx, info in enumerate(score["details"]):
                            status = "✅ Correct" if info["correct"] else "❌ Incorrect"
                            st.markdown(f"**Q{idx + 1}. {status}**")
                            st.write(info["question"])
                            st.markdown(
                                f"- **Correct answer:** {info['correct_option'] if info['correct_option'] is not None else 'N/A'}"
                            )
                            st.markdown(
                                f"- **Your answer:** {info['selected_option'] if info['selected_option'] is not None else 'No answer'}"
                            )
                            if info["explanation"]:
                                st.markdown(
                                    f"- **Explanation:** {info['explanation']}"
                                )
                            st.markdown("---")
if __name__ == "__main__":
    main()


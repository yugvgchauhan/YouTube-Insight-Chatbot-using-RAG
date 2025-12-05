"""
RAG Chain Module
"""
import os
import sys
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RAGChain:
    """Manages the RAG (Retrieval-Augmented Generation) chain."""

    def __init__(
        self,
        retriever,
        answer_language: str = "en",
        answer_tone: str = "neutral",
        answer_style: str = "auto",
        use_multi_query: bool = False,
    ):
        """
        Initialize RAG chain with configuration.

        Args:
            retriever: LangChain retriever instance
            answer_language: Target language for answers (e.g. 'en', 'hi')
            answer_tone: Desired tone (e.g. 'neutral', 'teacher', 'friendly')
            answer_style: Answer style (e.g. 'auto', 'concise', 'detailed')
            use_multi_query: Whether to use multi-query expansion for retrieval
        """
        self.retriever = retriever
        self.answer_language = answer_language
        self.answer_tone = answer_tone
        self.answer_style = answer_style
        self.use_multi_query = use_multi_query

        self.llm = self._initialize_llm()
        self.chain = self._build_chain()
    
    def _initialize_llm(self):
        """Initialize the HuggingFace LLM."""
        if not config.HUGGINGFACE_API_TOKEN:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")
        
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=config.HUGGINGFACE_LLM_MODEL,
            huggingfacehub_api_token=config.HUGGINGFACE_API_TOKEN,
            temperature=config.LLM_TEMPERATURE,
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
        )
        
        return ChatHuggingFace(llm=llm_endpoint)
    
    def _build_chain(self):
        """Build the RAG chain (LLM + prompt). Retrieval is handled explicitly per query."""
        # Define prompt template with configurable language, tone and style
        prompt_text = f"""
            You are a helpful assistant that answers questions based on YouTube video transcripts.
            Answer ONLY from the provided transcript context.
            If the context is insufficient to answer the question, politely say that you don't have enough information.

            You must answer in the language: {self.answer_language}.
            Your tone should be: {self.answer_tone}.
            Your answer style should be: {self.answer_style}.

            Context:
            {{context}}

            Question: {{question}}

            Answer:
            """

        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["context", "question"],
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _expand_queries(self, question: str):
        """Generate multiple related queries for better retrieval coverage."""
        instruction = (
            "Generate 3 alternative search queries that would help retrieve transcript "
            "segments relevant to answering the following question about a YouTube video. "
            "Return only the queries, one per line, without numbering or bullets.\n\n"
            f"Original question:\n{question}"
        )

        try:
            raw = self.llm.invoke(instruction)
            text = getattr(raw, "content", str(raw))
            lines = [ln.strip("-• ").strip() for ln in text.splitlines()]
            queries = [q for q in lines if q]
        except Exception:
            # Fall back to using only the original question
            queries = []

        # Always include the original question
        queries.append(question)

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        return unique_queries

    def _call_retriever(self, query: str):
        """Call the retriever in a version-tolerant way."""
        # Newer LangChain retrievers are Runnable, older ones expose get_relevant_documents
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(query)
        # Fallback: treat as runnable
        return self.retriever.invoke(query)

    def _get_relevant_docs(self, question: str):
        """Retrieve relevant documents, optionally using multi-query expansion."""
        if not self.use_multi_query:
            return self._call_retriever(question)

        # Multi-query expansion
        queries = self._expand_queries(question)
        all_docs = []
        seen_contents = set()

        for q in queries:
            try:
                docs = self._call_retriever(q)
            except Exception:
                continue

            # Ensure we always iterate over a list
            if not isinstance(docs, list):
                docs = [docs]

            for doc in docs:
                content = getattr(doc, "page_content", "")
                if content and content not in seen_contents:
                    seen_contents.add(content)
                    all_docs.append(doc)

        return all_docs

    def query(self, question: str) -> str:
        """
        Query the RAG chain with a question.
        
        Args:
            question: User's question
            
        Returns:
            Answer from the RAG chain
        """
        try:
            docs = self._get_relevant_docs(question)
            context = self._format_docs(docs)
            response = self.chain.invoke({"context": context, "question": question})
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def query_with_sources(self, question: str):
        """
        Query the RAG chain and also return the retrieved source documents.

        Args:
            question: User's question

        Returns:
            Tuple of (answer, source_documents)
        """
        try:
            source_docs = self._get_relevant_docs(question)
            context = self._format_docs(source_docs)
            answer = self.chain.invoke({"context": context, "question": question})
            return answer, source_docs
        except Exception as e:
            return f"Error generating response: {str(e)}", []


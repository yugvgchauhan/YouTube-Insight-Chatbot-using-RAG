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
    
    def __init__(self, retriever):
        """
        Initialize RAG chain with a retriever.
        
        Args:
            retriever: LangChain retriever instance
        """
        self.retriever = retriever
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
        """Build the RAG chain."""
        # Define prompt template
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant that answers questions based on YouTube video transcripts.
            Answer ONLY from the provided transcript context.
            If the context is insufficient to answer the question, politely say that you don't have enough information.
            Be concise and accurate in your responses.

            Context:
            {context}

            Question: {question}

            Answer:
            """,
            input_variables=["context", "question"]
        )
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Build parallel chain for context and question
        parallel_chain = RunnableParallel({
            "context": self.retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        
        # Build main chain
        chain = parallel_chain | prompt | self.llm | StrOutputParser()
        
        return chain
    
    def query(self, question: str) -> str:
        """
        Query the RAG chain with a question.
        
        Args:
            question: User's question
            
        Returns:
            Answer from the RAG chain
        """
        try:
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"


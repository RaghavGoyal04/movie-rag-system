"""
Main Movie RAG System - Refactored for better modularity
Orchestrates VectorStore and QA Chain managers
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI

from ..models.database import DatabaseManager
from .vectorstore_manager import VectorStoreManager
from .qa_chain_manager import QAChainManager

load_dotenv()


class MovieRAGSystem:
    """Main Movie RAG System - Orchestrates all components"""
    
    def __init__(self, 
                 database_url: str = None,
                 chroma_host: str = None,
                 chroma_port: int = None,
                 rebuild_vectorstore: bool = False):
        """
        Initialize Movie RAG System
        
        Args:
            database_url: PostgreSQL database URL
            chroma_host: ChromaDB host
            chroma_port: ChromaDB port
            rebuild_vectorstore: Whether to rebuild the vector store
        """
        
        # ChromaDB configuration
        self.chroma_host = chroma_host or os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = chroma_port or int(os.getenv("CHROMA_PORT", "8000"))
        
        # Initialize database
        if database_url is None:
            database_url = self._build_postgres_url()
        
        self.db = DatabaseManager(database_url)
        
        # Initialize LLM and embeddings
        self.embeddings, self.llm = self._initialize_ai_models()
        
        # Initialize managers
        self.vectorstore_manager = VectorStoreManager(
            embeddings=self.embeddings,
            chroma_host=self.chroma_host,
            chroma_port=self.chroma_port
        )
        
        # Setup vectorstore
        vectorstore = self.vectorstore_manager.setup_vectorstore(
            db_manager=self.db,
            rebuild=rebuild_vectorstore
        )
        
        if vectorstore:
            # Initialize QA chain manager
            retriever = self.vectorstore_manager.get_retriever()
            self.qa_manager = QAChainManager(
                llm=self.llm,
                retriever=retriever,
                db_manager=self.db
            )
        else:
            self.qa_manager = None
            print("⚠️ QA manager not initialized due to vectorstore setup failure")
    
    def _build_postgres_url(self) -> str:
        """Build PostgreSQL connection URL from environment variables"""
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5433")
        postgres_user = os.getenv("POSTGRES_USER", "postgres")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "movie_rag_password")
        postgres_db = os.getenv("POSTGRES_DB", "movie_rag_db")
        
        return f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    
    def _initialize_ai_models(self):
        """Initialize embeddings and LLM with GitHub models or OpenAI fallback"""
        github_token = os.getenv("GITHUB_TOKEN")
        github_base_url = os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference")
        embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        chat_model = os.getenv("CHAT_MODEL", "openai/gpt-4.1")
        
        if github_token:
            print(f"🔗 Using GitHub models - Embedding: {embedding_model}, Chat: {chat_model}")
            
            embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=github_token,
                openai_api_base=github_base_url
            )
            
            llm = ChatOpenAI(
                model=chat_model,
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
                openai_api_key=github_token,
                openai_api_base=github_base_url
            )
        else:
            # Fallback to OpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("Either GITHUB_TOKEN or OPENAI_API_KEY must be set")
            
            print("🔗 Using OpenAI models")
            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
            llm = OpenAI(temperature=0.7, openai_api_key=openai_key)
        
        return embeddings, llm
    
    # Delegate main functions to appropriate managers
    
    def get_recommendations(self, query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """Get movie recommendations"""
        if not self.qa_manager:
            raise RuntimeError("QA manager not initialized")
        return self.qa_manager.get_recommendations(query, user_id)
    
    def search_movies_advanced(self, **kwargs) -> List[Dict]:
        """Advanced movie search"""
        if not self.qa_manager:
            raise RuntimeError("QA manager not initialized")
        return self.qa_manager.search_movies_advanced(**kwargs)
    
    def get_similar_movies(self, movie_id: int, limit: int = 5) -> List[Dict]:
        """Get similar movies"""
        if not self.qa_manager:
            raise RuntimeError("QA manager not initialized")
        return self.qa_manager.get_similar_movies(movie_id, limit)
    
    def get_recommendations_by_mood(self, mood: str) -> List[Dict]:
        """Get recommendations by mood"""
        if not self.qa_manager:
            raise RuntimeError("QA manager not initialized")
        return self.qa_manager.get_recommendations_by_mood(mood)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        return self.db.get_movie_stats()
    
    def get_popular_searches(self, limit: int = 10) -> List[Dict]:
        """Get most popular search queries"""
        return self.db.get_popular_searches(limit)
    
    def rebuild_vectorstore(self):
        """Rebuild the vector store"""
        vectorstore = self.vectorstore_manager.setup_vectorstore(
            db_manager=self.db,
            rebuild=True
        )
        
        if vectorstore and self.qa_manager:
            # Update retriever in QA manager
            retriever = self.vectorstore_manager.get_retriever()
            self.qa_manager.update_retriever(retriever)
    
    # Getters for components
    
    def get_vectorstore_manager(self) -> VectorStoreManager:
        """Get vectorstore manager"""
        return self.vectorstore_manager
    
    def get_qa_manager(self) -> Optional[QAChainManager]:
        """Get QA chain manager"""
        return self.qa_manager
    
    def get_database_manager(self) -> DatabaseManager:
        """Get database manager"""
        return self.db


def main():
    """Test the refactored RAG system"""
    try:
        print("🎬 Initializing Refactored Movie RAG System...")
        
        # Display connection info
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5433")
        postgres_user = os.getenv("POSTGRES_USER", "postgres")
        postgres_db = os.getenv("POSTGRES_DB", "movie_rag_db")
        
        print("🗃️ Connecting to PostgreSQL database...")
        print(f"📍 Host: {postgres_host}:{postgres_port}")
        print(f"📍 Database: {postgres_db}")
        print(f"📍 User: {postgres_user}")
        
        # Initialize RAG system
        rag_system = MovieRAGSystem()
        
        # Test database features
        print("\n📊 Database Statistics:")
        stats = rag_system.get_database_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test advanced search
        print("\n🔍 Testing advanced search...")
        sci_fi_movies = rag_system.search_movies_advanced(
            genre="Science Fiction", 
            min_rating=8.0, 
            limit=3
        )
        
        print(f"Found {len(sci_fi_movies)} high-rated sci-fi movies:")
        for movie in sci_fi_movies:
            print(f"  • {movie['title']} ({movie['year']}) - ⭐ {movie['rating']}")
        
        # Test recommendations
        print("\n🤖 Testing AI recommendations...")
        test_queries = [
            "Recommend me mind-bending sci-fi movies like Inception",
            "What are some feel-good comedies from the 2000s?",
            "Dark psychological thrillers with unreliable narrators"
        ]
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            result = rag_system.get_recommendations(query)
            print(f"🤖 Answer preview: {result['answer'][:100]}...")
            print(f"📽️ Source movies: {[m['title'] for m in result['source_movies'][:3]]}")
            print(f"⚡ Response time: {result['response_time']:.2f}s")
            print("-" * 50)
        
        # Test mood-based recommendations
        print("\n🎭 Testing mood-based recommendations...")
        happy_movies = rag_system.get_recommendations_by_mood("happy")
        print(f"Happy mood movies: {[m['title'] for m in happy_movies[:3]]}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

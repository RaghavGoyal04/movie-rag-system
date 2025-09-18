#!/usr/bin/env python3
"""
End-to-End Test Script for Movie RAG System
Tests the complete pipeline from data collection to RAG recommendations
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import time

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

load_dotenv()

def test_environment():
    """Test environment variables and API keys"""
    print("🔧 Testing environment configuration...")
    
    # Check TMDB API key
    tmdb_key = os.getenv("MOVIE_API_KEY") or os.getenv("API_KEY")
    if tmdb_key:
        print(f"✅ TMDB API key found (ends with: ...{tmdb_key[-4:]})")
    else:
        print("❌ TMDB API key not found")
        return False
    
    # Check GitHub token
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        print(f"✅ GitHub token found (ends with: ...{github_token[-4:]})")
    else:
        print("⚠️  GitHub token not found")
    
    # Check OpenAI key (fallback)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OpenAI API key found (ends with: ...{openai_key[-4:]})")
    elif not github_token:
        print("❌ No AI provider keys found")
        return False
    
    return True

def test_database_connection():
    """Test PostgreSQL database connection"""
    print("\n🗃️ Testing database connection...")
    
    try:
        from movie_rag.models.database import DatabaseManager
        
        # Test database connection
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5433")
        postgres_user = os.getenv("POSTGRES_USER", "postgres")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "movie_rag_password")
        postgres_db = os.getenv("POSTGRES_DB", "movie_rag_db")
        
        postgres_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        
        db = DatabaseManager(postgres_url)
        stats = db.get_movie_stats()
        
        print(f"✅ Database connected successfully")
        print(f"   Movies: {stats.get('total_movies', 0)}")
        print(f"   Genres: {stats.get('total_genres', 0)}")
        print(f"   Average rating: {stats.get('avg_rating', 0):.1f}")
        
        return stats.get('total_movies', 0) > 0
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_chromadb_connection():
    """Test ChromaDB connection"""
    print("\n🧬 Testing ChromaDB connection...")
    
    try:
        import chromadb
        
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        heartbeat = client.heartbeat()
        
        print(f"✅ ChromaDB connected successfully at {chroma_host}:{chroma_port}")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return False

def test_rag_system():
    """Test the complete RAG system"""
    print("\n🤖 Testing RAG system...")
    
    try:
        from movie_rag.core.movie_rag_system import MovieRAGSystem
        
        print("🔄 Initializing RAG system...")
        rag_system = MovieRAGSystem()
        
        print("🧪 Testing recommendation...")
        test_query = "Recommend sci-fi movies with great visual effects"
        
        start_time = time.time()
        result = rag_system.get_recommendations(test_query)
        response_time = time.time() - start_time
        
        print(f"✅ RAG test successful!")
        print(f"   Query: {test_query}")
        print(f"   Response time: {response_time:.2f}s")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Source movies: {len(result['source_movies'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def show_next_steps():
    """Show next steps to user"""
    print("\n🚀 Next Steps:")
    print("   1. Docker services:")
    print("      docker-compose up -d")
    print("   2. Run CLI interface:")
    print("      uv run main.py")
    print("   3. Run web interface:")
    print("      uv run run_streamlit.py")
    print("   4. Access pgAdmin:")
    print("      http://localhost:5050")

def main():
    """Run all tests"""
    print("🎬 Movie RAG System - End-to-End Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test environment
    if not test_environment():
        all_passed = False
    
    # Test database
    if not test_database_connection():
        all_passed = False
    
    # Test ChromaDB
    if not test_chromadb_connection():
        all_passed = False
    
    # Test RAG system
    if not test_rag_system():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! The Movie RAG system is working correctly.")
    else:
        print("❌ Some tests failed. Please check the configuration.")
    
    show_next_steps()
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

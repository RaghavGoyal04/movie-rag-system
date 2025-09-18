#!/usr/bin/env python3
"""
Quick Test Script for Movie RAG System Components
Tests individual components quickly without full system initialization
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

load_dotenv()

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from movie_rag.models.database import DatabaseManager
        print("✅ database")
    except ImportError as e:
        print(f"❌ database: {e}")
        return False
    
    try:
        from movie_rag.data.movie_data_collector import TMDBDataCollector
        print("✅ data collector")
    except ImportError as e:
        print(f"❌ data collector: {e}")
        return False
    
    try:
        from movie_rag.core.movie_rag_enhanced import MovieRAGSystem
        print("✅ RAG system")
    except ImportError as e:
        print(f"❌ RAG system: {e}")
        return False
    
    return True

def test_database():
    """Test database connection"""
    print("\n🧪 Testing database...")
    
    try:
        from movie_rag.models.database import DatabaseManager
        
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5433")
        postgres_user = os.getenv("POSTGRES_USER", "postgres")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "movie_rag_password")
        postgres_db = os.getenv("POSTGRES_DB", "movie_rag_db")
        
        postgres_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        
        db = DatabaseManager(postgres_url)
        stats = db.get_movie_stats()
        
        print(f"✅ Database connection successful: {stats.get('total_movies', 0)} movies")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_llm_direct():
    """Test LLM connection directly"""
    print("\n🧪 Testing LLM connection...")
    
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        github_base_url = os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference")
        chat_model = os.getenv("CHAT_MODEL", "openai/gpt-4.1")
        
        if github_token:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model=chat_model,
                temperature=0.7,
                max_tokens=500,
                openai_api_key=github_token,
                openai_api_base=github_base_url
            )
            
            test_response = llm.invoke("Recommend a good action movie in one sentence.")
            print(f"✅ LLM test successful: {test_response.content[:100]}...")
            return True
        else:
            print("⚠️ No GitHub token found, skipping LLM test")
            return False
            
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

def main():
    """Run quick tests"""
    print("🎬 Movie RAG System - Quick Component Test")
    print("=" * 50)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_database():
        all_passed = False
    
    if not test_llm_direct():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All quick tests passed!")
    else:
        print("❌ Some tests failed.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

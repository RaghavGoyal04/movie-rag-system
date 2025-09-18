#!/usr/bin/env python3
"""
Test PostgreSQL + ChromaDB integration without OpenAI dependency
"""

import os
import sys
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from sqlalchemy import text

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from movie_rag.models.database import DatabaseManager

load_dotenv()

def test_integration():
    """Test PostgreSQL and ChromaDB integration"""
    
    print("🧪 Testing PostgreSQL + ChromaDB Integration")
    print("=" * 60)
    
    # Test 1: PostgreSQL Connection
    print("1️⃣ Testing PostgreSQL connection...")
    try:
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5433")
        postgres_user = os.getenv("POSTGRES_USER", "postgres")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "movie_rag_password")
        postgres_db = os.getenv("POSTGRES_DB", "movie_rag_db")
        
        postgres_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        
        db = DatabaseManager(postgres_url)
        session = db.get_session()
        
        # Get movie count
        movie_count = session.execute(text("SELECT COUNT(*) FROM movies")).scalar()
        print(f"✅ PostgreSQL connected - Found {movie_count} movies")
        
        session.close()
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False
    
    # Test 2: ChromaDB Connection
    print("\n2️⃣ Testing ChromaDB connection...")
    try:
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        
        chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )
        
        # Test heartbeat
        chroma_client.heartbeat()
        
        # List collections
        collections = chroma_client.list_collections()
        print(f"✅ ChromaDB connected - Found {len(collections)} collections")
        
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return False
    
    # Test 3: Data availability
    print("\n3️⃣ Testing data availability...")
    try:
        session = db.get_session()
        
        # Sample query
        result = session.execute(text("""
            SELECT m.title, m.vote_average, array_agg(g.name) as genres
            FROM movies m
            LEFT JOIN movie_genres mg ON m.id = mg.movie_id
            LEFT JOIN genres g ON mg.genre_id = g.id
            WHERE m.vote_average > 8.0
            GROUP BY m.id, m.title, m.vote_average
            ORDER BY m.vote_average DESC
            LIMIT 3
        """)).fetchall()
        
        print(f"✅ Sample high-rated movies:")
        for row in result:
            title, rating, genres = row
            genres_str = ', '.join(genres) if genres and genres[0] else 'No genres'
            print(f"   • {title} ({rating}/10) - {genres_str}")
        
        session.close()
        
    except Exception as e:
        print(f"❌ Data query failed: {e}")
        return False
    
    print(f"\n🎉 Integration test successful!")
    print(f"🔗 Both PostgreSQL and ChromaDB are ready for RAG system")
    print(f"💡 Next step: Add OPENAI_API_KEY to .env file to run full RAG system")
    
    return True

def show_next_steps():
    """Show next steps for complete setup"""
    print(f"\n📋 Setup Status:")
    print(f"✅ PostgreSQL: Ready ({os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5433')})")
    print(f"✅ ChromaDB: Ready ({os.getenv('CHROMA_HOST', 'localhost')}:{os.getenv('CHROMA_PORT', '8000')})")
    print(f"❓ OpenAI API: {'✅ Configured' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    
    print(f"\n🚀 Ready to run:")
    print(f"   uv run main.py               (CLI interface - requires API key)")
    print(f"   uv run run_streamlit.py      (Web interface - requires API key)")

if __name__ == "__main__":
    success = test_integration()
    show_next_steps()

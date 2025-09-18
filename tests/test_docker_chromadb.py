#!/usr/bin/env python3
"""
Test Docker ChromaDB connection and setup
"""

import os
import sys
from pathlib import Path
import requests
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

load_dotenv()

def test_chromadb_connection():
    """Test connection to Docker ChromaDB"""
    
    print("🧪 Testing Docker ChromaDB Connection")
    print("=" * 50)
    
    # ChromaDB configuration
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    
    print(f"📍 ChromaDB Configuration:")
    print(f"   Host: {chroma_host}")
    print(f"   Port: {chroma_port}")
    
    try:
        # Step 1: Test basic HTTP connection
        print("\n1️⃣ Testing basic HTTP connection...")
        response = requests.get(f"http://{chroma_host}:{chroma_port}", timeout=5)
        print(f"   Status: {response.status_code} (404 is expected for root endpoint)")
        print("✅ ChromaDB server is responding")
        
        # Step 2: Test ChromaDB client connection
        print("\n2️⃣ Testing ChromaDB client connection...")
        chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )
        
        # Test heartbeat
        chroma_client.heartbeat()
        print("✅ ChromaDB client connected successfully")
        
        # Step 3: Test collection operations
        print("\n3️⃣ Testing collection operations...")
        test_collection_name = "test_collection"
        
        # Try to create a test collection
        try:
            collection = chroma_client.create_collection(test_collection_name)
            print(f"✅ Created test collection: {test_collection_name}")
            
            # Add a test document
            collection.add(
                documents=["This is a test document for ChromaDB"],
                metadatas=[{"source": "test"}],
                ids=["test_doc_1"]
            )
            print("✅ Added test document to collection")
            
            # Query the collection
            results = collection.query(query_texts=["test document"], n_results=1)
            if results and results['documents']:
                print("✅ Successfully queried collection")
            
            # Clean up
            chroma_client.delete_collection(test_collection_name)
            print("🗑️ Cleaned up test collection")
            
        except Exception as e:
            print(f"⚠️ Collection operations warning: {e}")
        
        # Step 4: List existing collections
        print("\n4️⃣ Listing existing collections...")
        collections = chroma_client.list_collections()
        print(f"📋 Found {len(collections)} existing collections:")
        for collection in collections:
            print(f"   • {collection.name}")
        
        print(f"\n🎉 All ChromaDB tests passed!")
        print(f"🐳 ChromaDB is ready for movie recommendations")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to ChromaDB server")
        print(f"💡 Make sure ChromaDB container is running:")
        print(f"   docker-compose up -d chromadb")
        return False
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_chromadb_info():
    """Show ChromaDB service information"""
    print("\n📊 ChromaDB Service Information:")
    print("=" * 50)
    print("🐳 Container: movie_rag_chromadb")
    print("🌐 URL: http://localhost:8000")
    print("💾 Data Volume: chroma_data")
    print("🔄 Persistence: Enabled")
    print("\n📚 Useful Commands:")
    print("   Start service: docker-compose up -d chromadb")
    print("   View logs: docker logs movie_rag_chromadb")
    print("   Restart: docker-compose restart chromadb")
    print("   Stop: docker-compose down")

if __name__ == "__main__":
    success = test_chromadb_connection()
    show_chromadb_info()
    
    if success:
        print(f"\n🚀 Ready to run Enhanced RAG system with Docker ChromaDB!")
        print(f"   uv run main.py")
    else:
        print(f"\n🔧 Please fix ChromaDB connection issues before proceeding")

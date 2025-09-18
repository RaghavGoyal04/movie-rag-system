"""
VectorStore Manager for Movie RAG System
Handles ChromaDB operations and document processing
"""

import os
from math import ceil
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import chromadb
from sqlalchemy.orm import joinedload

from ..models.database import DatabaseManager, Movie


class VectorStoreManager:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self, 
                 embeddings: OpenAIEmbeddings,
                 chroma_host: str = "localhost", 
                 chroma_port: int = 8000):
        """
        Initialize VectorStore Manager
        
        Args:
            embeddings: OpenAI embeddings instance
            chroma_host: ChromaDB host
            chroma_port: ChromaDB port
        """
        self.embeddings = embeddings
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.vectorstore = None
        self.chroma_client = None
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )
    
    def connect_to_chromadb(self) -> chromadb.HttpClient:
        """Connect to ChromaDB server"""
        print(f"🐳 Connecting to ChromaDB server at {self.chroma_host}:{self.chroma_port}...")
        
        try:
            self.chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port
            )
            
            # Test connection
            self.chroma_client.heartbeat()
            print("✅ Connected to ChromaDB server successfully")
            return self.chroma_client
            
        except Exception as e:
            print(f"❌ Failed to connect to ChromaDB server: {e}")
            print("💡 Make sure ChromaDB container is running: docker-compose up -d chromadb")
            raise
    
    def setup_vectorstore(self, 
                         db_manager: DatabaseManager, 
                         collection_name: str = "movie_recommendations",
                         rebuild: bool = False) -> Chroma:
        """
        Setup or load vector store from database
        
        Args:
            db_manager: Database manager instance
            collection_name: Name of ChromaDB collection
            rebuild: Whether to rebuild existing collection
        
        Returns:
            Chroma vectorstore instance
        """
        if not self.chroma_client:
            self.connect_to_chromadb()
        
        # Check if collection exists and load if not rebuilding
        if not rebuild:
            try:
                collection = self.chroma_client.get_collection(collection_name)
                print(f"📁 Loading existing collection '{collection_name}'...")
                self.vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings
                )
                print(f"✅ Loaded existing vectorstore with {collection.count()} documents")
                return self.vectorstore
            except Exception:
                print(f"📝 Collection '{collection_name}' doesn't exist, will create new one")
        else:
            try:
                print(f"🗑️ Deleting existing collection for rebuild...")
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                pass  # Collection doesn't exist
        
        # Create new vectorstore
        return self._create_new_vectorstore(db_manager, collection_name)
    
    def _create_new_vectorstore(self, 
                              db_manager: DatabaseManager, 
                              collection_name: str) -> Chroma:
        """Create new vectorstore from database data"""
        print("🔨 Creating new vectorstore from database...")
        
        # Get all movies with relationships
        movies = self._load_movies_from_database(db_manager)
        
        if not movies:
            print("❌ No movies found in database. Please load movies first.")
            return None
        
        print(f"📊 Processing {len(movies)} movies from database...")
        
        # Process movies into documents
        documents = self._process_movies_to_documents(movies)
        
        # Split documents into chunks
        docs = self._split_documents_into_chunks(documents)
        
        # Create and populate vectorstore
        return self._create_and_populate_vectorstore(docs, collection_name)
    
    def _load_movies_from_database(self, db_manager: DatabaseManager) -> List[Movie]:
        """Load movies from database with eager loading"""
        session = db_manager.get_session()
        
        try:
            movies = session.query(Movie).options(
                joinedload(Movie.genres),
                joinedload(Movie.actors),
                joinedload(Movie.directors)
            ).all()
            return movies
        finally:
            session.close()
    
    def _process_movies_to_documents(self, movies: List[Movie]) -> List[Dict]:
        """Convert movies to document format with metadata"""
        documents = []
        
        for movie in movies:
            doc_text = self._create_movie_document_from_db(movie)
            
            metadata = {
                'movie_id': movie.id,
                'title': movie.title,
                'release_date': movie.release_date or '',
                'genres': ', '.join([g.name for g in movie.genres]),
                'vote_average': movie.vote_average or 0,
                'popularity': movie.popularity or 0,
                'runtime': movie.runtime or 0,
                'budget': movie.budget or 0,
                'revenue': movie.revenue or 0,
                'directors': ', '.join([d.name for d in movie.directors]),
                'actors': ', '.join([a.name for a in movie.actors[:5]])
            }
            
            documents.append({
                'text': doc_text,
                'metadata': metadata
            })
        
        return documents
    
    def _create_movie_document_from_db(self, movie: Movie) -> str:
        """Create comprehensive text document from database movie object"""
        
        # Basic info
        title = movie.title or 'Unknown Title'
        release_date = movie.release_date or 'Unknown'
        overview = movie.overview or 'No overview available'
        tagline = movie.tagline or ''
        
        # Ratings and popularity
        vote_average = movie.vote_average or 0
        vote_count = movie.vote_count or 0
        popularity = movie.popularity or 0
        
        # Genres
        genres = [genre.name for genre in movie.genres]
        genre_text = ', '.join(genres) if genres else 'Unknown'
        
        # Cast (top 5 actors)
        cast_names = [actor.name for actor in movie.actors[:5]]
        cast_text = ', '.join(cast_names) if cast_names else 'Unknown cast'
        
        # Directors
        directors = [director.name for director in movie.directors]
        director_text = ', '.join(directors) if directors else 'Unknown director'
        
        # Production companies
        companies = [company.name for company in movie.companies]
        companies_text = ', '.join(companies) if companies else ''
        
        # Keywords
        keywords = movie.keywords.get('keywords', []) if movie.keywords else []
        keyword_names = [kw.get('name', '') for kw in keywords]
        keywords_text = ', '.join(keyword_names) if keyword_names else ''
        
        # Create comprehensive document
        document_text = f"""
        Movie Title: {title}
        Release Date: {release_date}
        Genres: {genre_text}
        Director: {director_text}
        Main Cast: {cast_text}
        Rating: {vote_average}/10 ({vote_count} votes)
        Popularity Score: {popularity}
        Runtime: {movie.runtime or 'Unknown'} minutes
        Budget: ${movie.budget or 0:,}
        Revenue: ${movie.revenue or 0:,}
        
        Plot Summary: {overview}
        
        Tagline: {tagline}
        
        Keywords: {keywords_text}
        Production Companies: {companies_text}
        
        Movie Analysis:
        This is a {genre_text.lower()} film directed by {director_text}, starring {cast_names[0] if cast_names else 'unknown actors'}. 
        Released in {release_date[:4] if release_date and len(release_date) >= 4 else 'unknown year'}, 
        it has received a rating of {vote_average}/10 from {vote_count} viewers.
        
        The story revolves around: {overview}
        
        Similar themes and elements: {keywords_text}
        
        Box Office Performance: With a budget of ${movie.budget or 0:,}, the film earned ${movie.revenue or 0:,} worldwide.
        
        Critical Reception: This {genre_text.lower()} film has achieved a popularity score of {popularity:.1f} and maintains 
        an average rating of {vote_average}/10 based on {vote_count} user votes.
        """
        
        return document_text.strip()
    
    def _split_documents_into_chunks(self, documents: List[Dict]) -> List[Document]:
        """Split documents into chunks and create Document objects"""
        print(f"📝 Creating document chunks...")
        
        docs = []
        for doc_data in documents:
            chunks = self.text_splitter.split_text(doc_data['text'])
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk,
                    metadata=doc_data['metadata']
                ))
        
        print(f"✂️ Created {len(docs)} document chunks")
        return docs
    
    def _create_and_populate_vectorstore(self, 
                                       docs: List[Document], 
                                       collection_name: str) -> Chroma:
        """Create vectorstore and populate with documents in batches"""
        print(f"🔄 Creating vectorstore in batches to avoid token limits...")
        
        # Create empty collection first
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Process documents in smaller batches
        batch_size = 50
        total_batches = ceil(len(docs) / batch_size)
        
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            batch_num = i // batch_size + 1
            print(f"   Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)")
            
            try:
                self.vectorstore.add_documents(batch_docs)
                print(f"   ✅ Batch {batch_num} added successfully")
            except Exception as e:
                print(f"   ⚠️ Batch {batch_num} failed: {e}")
                # Continue with next batch
        
        print(f"✅ Created and stored vectorstore in ChromaDB with {len(docs)} chunks")
        return self.vectorstore
    
    def get_vectorstore(self) -> Optional[Chroma]:
        """Get the vectorstore instance"""
        return self.vectorstore
    
    def get_retriever(self, search_kwargs: Dict = None):
        """Get retriever from vectorstore"""
        if not self.vectorstore:
            raise RuntimeError("Vectorstore not initialized")
        
        search_kwargs = search_kwargs or {"k": 8}
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs # Return top 8 results
        )

"""
 Movie RAG System with Database Integration
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
from ..models.database import DatabaseManager, Movie, Genre, Actor, Director

load_dotenv()

class MovieRAGSystem:
    """ RAG-based Movie Recommendation System with Database"""
    
    def __init__(self, 
                 database_url: str = None,
                 chroma_host: str = None,
                 chroma_port: int = None,
                 rebuild_vectorstore: bool = False):
        
        self.vectorstore = None
        self.qa_chain = None
        
        # ChromaDB configuration
        self.chroma_host = chroma_host or os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = chroma_port or int(os.getenv("CHROMA_PORT", "8000"))
        
        # Initialize database
        if database_url is None:
            database_url = self._build_postgres_url()
        
        self.db = DatabaseManager(database_url)
        
        # Initialize embeddings and LLM with GitHub models
        github_token = os.getenv("GITHUB_TOKEN")
        github_base_url = os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference")
        embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        chat_model = os.getenv("CHAT_MODEL", "openai/gpt-4.1")
        
        if github_token:
            print(f"🔗 Using GitHub models - Embedding: {embedding_model}, Chat: {chat_model}")
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=github_token,
                openai_api_base=github_base_url
            )
            self.llm = ChatOpenAI(
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
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
            self.llm = OpenAI(temperature=0.7, openai_api_key=openai_key)
        
        # Setup components
        self.setup_vectorstore(rebuild=rebuild_vectorstore)
        self.setup_qa_chain()
    
    def _build_postgres_url(self) -> str:
        """
        Build PostgreSQL connection URL from environment variables
        
        Returns:
            str: Complete PostgreSQL connection URL
            
        Environment Variables:
            POSTGRES_HOST: Database host (default: localhost)
            POSTGRES_PORT: Database port (default: 5433)
            POSTGRES_USER: Database username (default: postgres)
            POSTGRES_PASSWORD: Database password (default: movie_rag_password)
            POSTGRES_DB: Database name (default: movie_rag_db)
        """
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5433")
        postgres_user = os.getenv("POSTGRES_USER", "postgres")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "movie_rag_password")
        postgres_db = os.getenv("POSTGRES_DB", "movie_rag_db")
        
        return f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    
    def create_movie_document_from_db(self, movie: Movie) -> str:
        """Create a comprehensive text document from database movie object"""
        
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
    
    def setup_vectorstore(self, rebuild: bool = False):
        """Create or load vector store with database movie data using Docker ChromaDB"""
        
        # Connect to Docker ChromaDB server
        print(f"🐳 Connecting to ChromaDB server at {self.chroma_host}:{self.chroma_port}...")
        
        try:
            # Create ChromaDB client for Docker container
            chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port
            )
            
            # Test connection
            chroma_client.heartbeat()
            print("✅ Connected to ChromaDB server successfully")
            
            # Check if collection already exists and we don't want to rebuild
            collection_name = "movie_recommendations"
            try:
                collection = chroma_client.get_collection(collection_name)
                if not rebuild:
                    print(f"📁 Loading existing collection '{collection_name}'...")
                    self.vectorstore = Chroma(
                        client=chroma_client,
                        collection_name=collection_name,
                        embedding_function=self.embeddings
                    )
                    print(f"✅ Loaded existing vectorstore with {collection.count()} documents")
                    return
                else:
                    print(f"🗑️ Deleting existing collection for rebuild...")
                    chroma_client.delete_collection(collection_name)
            except Exception:
                print(f"📝 Collection '{collection_name}' doesn't exist, will create new one")
                
        except Exception as e:
            print(f"❌ Failed to connect to ChromaDB server: {e}")
            print("💡 Make sure ChromaDB container is running: docker-compose up -d chromadb")
            raise
        
        print("🔨 Creating new vectorstore from database...")
        
        # Get all movies from database with eager loading
        session = self.db.get_session()
        from sqlalchemy.orm import joinedload
        movies = session.query(Movie).options(
            joinedload(Movie.genres),
            joinedload(Movie.actors),
            joinedload(Movie.directors)
        ).all()
        # Keep session open until we're done processing movies
        
        if not movies:
            print("❌ No movies found in database. Please load movies first.")
            return
        
        print(f"📊 Processing {len(movies)} movies from database...")
        
        # Create documents for each movie
        documents = []
        metadatas = []
        
        for movie in movies:
            doc_text = self.create_movie_document_from_db(movie)
            
            # Create metadata
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
            
            documents.append(doc_text)
            metadatas.append(metadata)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        print(f"📝 Creating document chunks...")
        
        # Create Document objects
        docs = []
        for i, (doc_text, metadata) in enumerate(zip(documents, metadatas)):
            chunks = text_splitter.split_text(doc_text)
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))
        
        print(f"✂️ Created {len(docs)} document chunks")
        
        # Close the database session now that we're done processing
        session.close()
        
        # Create vectorstore using Docker ChromaDB with batch processing
        print(f"🔄 Creating vectorstore in batches to avoid token limits...")
        
        # Create empty collection first
        self.vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Process documents in smaller batches to avoid token limits
        batch_size = 50  # Smaller batch size for GitHub models
        total_batches = (len(docs) + batch_size - 1) // batch_size
        
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
    
    def setup_qa_chain(self):
        """Setup the QA chain with  prompt"""
        
        #  prompt for movie recommendations
        prompt_template = """
        You are CineGenie, an expert movie recommendation assistant with access to a comprehensive movie database. 
        Use the following movie information to provide detailed, personalized recommendations.

        Context: {context}

        Question: {question}

        Instructions:
        - Provide specific movie recommendations with titles, years, ratings, and detailed explanations
        - Include director, main cast, and genre information when relevant
        - Explain WHY each movie fits the user's request with specific details
        - If recommending multiple movies, rank them by relevance
        - Include budget/revenue information for blockbusters when relevant
        - Mention similar themes, plot elements, or cinematic techniques
        - Always cite the source information and be accurate with facts
        - If you don't have enough information, suggest related queries or criteria

        Response Format:
        - Start with a brief summary of what the user is looking for
        - List 3-5 specific movie recommendations
        - For each movie, include: Title (Year), Director, Rating, Brief explanation
        - End with additional suggestions or related recommendations

        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}  # Retrieve more context for better recommendations
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✅  QA Chain setup complete")
    
    def get_recommendations(self, query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """Get movie recommendations with  features"""
        
        if not self.qa_chain:
            raise RuntimeError("QA chain not initialized.")
        
        start_time = time.time()
        
        # Get response from QA chain
        response = self.qa_chain.invoke({"query": query})
        
        response_time = time.time() - start_time
        
        # Extract source movies with database enhancement
        source_movies = []
        for doc in response.get('source_documents', []):
            metadata = doc.metadata
            movie_id = metadata.get('movie_id')
            
            # Get full movie details from database
            if movie_id:
                movie = self.db.get_movie_by_id(movie_id)
                if movie:
                    source_movies.append({
                        'id': movie.id,
                        'title': movie.title,
                        'year': movie.release_date[:4] if movie.release_date else '',
                        'rating': movie.vote_average,
                        'genres': [g.name for g in movie.genres],
                        'directors': [d.name for d in movie.directors],
                        'actors': [a.name for a in movie.actors[:3]],
                        'overview': movie.overview[:200] + '...' if movie.overview else '',
                        'popularity': movie.popularity,
                        'runtime': movie.runtime
                    })
        
        # Remove duplicates
        unique_movies = []
        seen_ids = set()
        for movie in source_movies:
            if movie['id'] not in seen_ids:
                unique_movies.append(movie)
                seen_ids.add(movie['id'])
        
        # Log search for analytics
        self.db.log_search(user_id, query, response_time, len(unique_movies))
        
        return {
            'answer': response['result'],
            'source_movies': unique_movies[:8],  # Top 8 source movies
            'query': query,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def search_movies_advanced(self, **criteria) -> List[Dict]:
        """Advanced movie search using database with multiple filters"""
        
        movies = self.db.search_movies(**criteria)
        
        result = []
        for movie in movies:
            result.append({
                'id': movie.id,
                'title': movie.title,
                'year': movie.release_date[:4] if movie.release_date else '',
                'rating': movie.vote_average,
                'genres': [g.name for g in movie.genres],
                'directors': [d.name for d in movie.directors],
                'actors': [a.name for a in movie.actors[:5]],
                'overview': movie.overview,
                'runtime': movie.runtime,
                'budget': movie.budget,
                'revenue': movie.revenue,
                'popularity': movie.popularity
            })
        
        return result
    
    def get_movie_details_(self, movie_id: int) -> Optional[Dict]:
        """Get comprehensive movie details from database"""
        
        movie = self.db.get_movie_by_id(movie_id)
        if not movie:
            return None
        
        return {
            'id': movie.id,
            'title': movie.title,
            'original_title': movie.original_title,
            'year': movie.release_date[:4] if movie.release_date else '',
            'release_date': movie.release_date,
            'rating': movie.vote_average,
            'vote_count': movie.vote_count,
            'genres': [g.name for g in movie.genres],
            'directors': [{'name': d.name, 'id': d.id} for d in movie.directors],
            'actors': [{'name': a.name, 'id': a.id} for a in movie.actors[:10]],
            'overview': movie.overview,
            'tagline': movie.tagline,
            'runtime': movie.runtime,
            'budget': movie.budget,
            'revenue': movie.revenue,
            'popularity': movie.popularity,
            'production_companies': [c.name for c in movie.companies],
            'keywords': movie.keywords,
            'spoken_languages': movie.spoken_languages,
            'production_countries': movie.production_countries,
            'homepage': movie.homepage,
            'imdb_id': movie.imdb_id
        }
    
    def get_similar_movies(self, movie_id: int, limit: int = 5) -> List[Dict]:
        """Find similar movies using vector similarity"""
        
        movie = self.db.get_movie_by_id(movie_id)
        if not movie:
            return []
        
        # Create query based on movie characteristics
        query = f"Movies similar to {movie.title} with {', '.join([g.name for g in movie.genres])} themes"
        
        # Use vector search to find similar movies
        docs = self.vectorstore.similarity_search(query, k=limit+5)  # Get extra to filter out the same movie
        
        similar_movies = []
        for doc in docs:
            similar_id = doc.metadata.get('movie_id')
            if similar_id and similar_id != movie_id:
                similar_movie = self.db.get_movie_by_id(similar_id)
                if similar_movie and len(similar_movies) < limit:
                    similar_movies.append({
                        'id': similar_movie.id,
                        'title': similar_movie.title,
                        'year': similar_movie.release_date[:4] if similar_movie.release_date else '',
                        'rating': similar_movie.vote_average,
                        'genres': [g.name for g in similar_movie.genres],
                        'overview': similar_movie.overview[:150] + '...' if similar_movie.overview else ''
                    })
        
        return similar_movies
    
    def get_recommendations_by_mood(self, mood: str) -> List[Dict]:
        """Get movie recommendations based on mood"""
        
        mood_queries = {
            'happy': "Feel-good comedy movies with uplifting themes and happy endings",
            'sad': "Emotional drama movies that are cathartic and meaningful",
            'excited': "Action-packed thriller movies with high energy and excitement",
            'relaxed': "Calm peaceful movies with beautiful cinematography and slow pace",
            'romantic': "Romantic movies with love stories and chemistry between leads",
            'scared': "Horror movies with suspense and supernatural elements",
            'thoughtful': "Philosophical movies that make you think about life and existence",
            'nostalgic': "Classic movies from the past with timeless appeal"
        }
        
        query = mood_queries.get(mood.lower(), f"Movies that match {mood} mood")
        result = self.get_recommendations(query)
        
        return result['source_movies']
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        return self.db.get_movie_stats()
    
    def get_popular_searches(self, limit: int = 10) -> List[Dict]:
        """Get most popular search queries"""
        return self.db.get_popular_searches(limit)

def main():
    """Test the  RAG system"""
    try:
        print("🎬 Initializing  Movie RAG System...")
        
        # Display connection info
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5433")
        postgres_user = os.getenv("POSTGRES_USER", "postgres")
        postgres_db = os.getenv("POSTGRES_DB", "movie_rag_db")
        
        print("🗃️ Connecting to PostgreSQL database...")
        print(f"📍 Host: {postgres_host}:{postgres_port}")
        print(f"📍 Database: {postgres_db}")
        print(f"📍 User: {postgres_user}")
        
        # Initialize RAG system (will auto-build PostgreSQL URL from env vars)
        rag_system = MovieRAGSystem()
        
        # Test  features
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
        print("\n🤖 Testing  recommendations...")
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
        
    except Exception as e:
        print(f"❌ Error testing  RAG system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

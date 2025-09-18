"""
QA Chain Manager for Movie RAG System
Handles LLM configuration and QA chain setup
"""

import time
from typing import Dict, Any, List
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI

from ..models.database import DatabaseManager


class QAChainManager:
    """Manages QA chain and LLM operations"""
    
    def __init__(self, llm, retriever, db_manager: DatabaseManager):
        """
        Initialize QA Chain Manager
        
        Args:
            llm: Language model instance (OpenAI or ChatOpenAI)
            retriever: Vectorstore retriever
            db_manager: Database manager for enhanced responses
        """
        self.llm = llm
        self.retriever = retriever
        self.db_manager = db_manager
        self.qa_chain = None
        
        # Setup QA chain on initialization
        self.setup_qa_chain()
    
    def setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        print("🔗 Setting up QA chain...")
        
        # Custom prompt template for movie recommendations
        prompt_template = """
        You are CineGenie, an AI movie recommendation expert. Use the following movie information to provide personalized, engaging recommendations.

        Context: {context}

        User Question: {question}

        Instructions:
        1. Provide specific movie recommendations based on the context
        2. Include brief explanations of why each movie fits the user's request
        3. Mention key details like genre, director, cast, or plot elements
        4. Be conversational and enthusiastic about movies
        5. If asked about specific movies, provide detailed information
        6. Always base your recommendations on the provided context

        Response:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✅ QA Chain setup complete")
    
    def get_recommendations(self, query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """
        Get movie recommendations with enhanced features
        
        Args:
            query: User query for recommendations
            user_id: User identifier for analytics
            
        Returns:
            Dict containing answer, source movies, and metadata
        """
        if not self.qa_chain:
            raise RuntimeError("QA chain not initialized.")
        
        start_time = time.time()
        
        # Get response from QA chain
        response = self.qa_chain.invoke({"query": query})
        
        response_time = time.time() - start_time
        
        # Extract and enhance source movies with database info
        source_movies = self._extract_enhanced_source_movies(response)
        
        # Log search for analytics
        self.db_manager.log_search(user_id, query, response_time, len(source_movies))
        
        return {
            'answer': response['result'],
            'source_movies': source_movies,
            'query': query,
            'response_time': response_time,
            'timestamp': time.time()
        }
    
    def _extract_enhanced_source_movies(self, response: Dict) -> List[Dict]:
        """Extract source movies and enhance with database information"""
        source_movies = []
        
        for doc in response.get('source_documents', []):
            metadata = doc.metadata
            movie_id = metadata.get('movie_id')
            
            # Get full movie details from database if available
            if movie_id:
                movie = self.db_manager.get_movie_by_id(movie_id)
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
        
        return unique_movies
    
    def search_movies_advanced(self, 
                             genre: str = None, 
                             min_rating: float = None,
                             max_rating: float = None,
                             year: int = None,
                             director: str = None,
                             actor: str = None,
                             limit: int = 10) -> List[Dict]:
        """
        Advanced movie search using database filters
        
        Args:
            genre: Movie genre to filter by
            min_rating: Minimum rating threshold
            max_rating: Maximum rating threshold
            year: Release year
            director: Director name
            actor: Actor name
            limit: Maximum number of results
            
        Returns:
            List of movies matching criteria
        """
        movies = self.db_manager.search_movies(
            genre=genre,
            min_rating=min_rating,
            max_rating=max_rating,
            year=year,
            director=director,
            actor=actor,
            limit=limit
        )
        
        return [{
            'id': movie.id,
            'title': movie.title,
            'year': movie.release_date[:4] if movie.release_date else 'Unknown',
            'rating': movie.vote_average or 0,
            'genres': [g.name for g in movie.genres],
            'directors': [d.name for d in movie.directors],
            'actors': [a.name for a in movie.actors[:3]],
            'overview': movie.overview,
            'popularity': movie.popularity,
            'runtime': movie.runtime
        } for movie in movies]
    
    def get_similar_movies(self, movie_id: int, limit: int = 5) -> List[Dict]:
        """
        Get movies similar to a specific movie using vector similarity
        
        Args:
            movie_id: ID of the reference movie
            limit: Number of similar movies to return
            
        Returns:
            List of similar movies
        """
        # Get the reference movie
        reference_movie = self.db_manager.get_movie_by_id(movie_id)
        if not reference_movie:
            return []
        
        # Create a query based on the movie's characteristics
        query = f"Movies similar to {reference_movie.title} with {', '.join([g.name for g in reference_movie.genres])} themes"
        
        # Get recommendations using the RAG system
        result = self.get_recommendations(query)
        
        # Filter out the reference movie and return similar ones
        similar_movies = []
        for movie in result['source_movies']:
            if movie['id'] != movie_id:
                similar_movies.append({
                    'id': movie['id'],
                    'title': movie['title'],
                    'year': movie['year'],
                    'rating': movie['rating'],
                    'genres': movie['genres'],
                    'directors': movie['directors'],
                    'actors': movie['actors'],
                    'overview': movie['overview'],
                    'similarity_reason': f"Similar themes and style to {reference_movie.title}"
                })
                
            if len(similar_movies) >= limit:
                break
        
        return similar_movies
    
    def get_recommendations_by_mood(self, mood: str) -> List[Dict]:
        """
        Get movie recommendations based on mood
        
        Args:
            mood: User's current mood
            
        Returns:
            List of movies matching the mood
        """
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
    
    def update_retriever(self, new_retriever):
        """Update the retriever and reinitialize QA chain"""
        self.retriever = new_retriever
        self.setup_qa_chain()
    
    def get_qa_chain(self):
        """Get the QA chain instance"""
        return self.qa_chain

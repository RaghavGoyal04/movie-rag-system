"""
Database models and operations for the Movie RAG system
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Table, ForeignKey, Boolean, BigInteger, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import JSON
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

# Association tables for many-to-many relationships
movie_genres = Table(
    'movie_genres',
    Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id'), primary_key=True),
    Column('genre_id', Integer, ForeignKey('genres.id'), primary_key=True)
)

movie_actors = Table(
    'movie_actors',
    Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id'), primary_key=True),
    Column('actor_id', Integer, ForeignKey('actors.id'), primary_key=True),
    Column('character_name', String(255)),
    Column('order', Integer)
)

movie_directors = Table(
    'movie_directors',
    Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id'), primary_key=True),
    Column('director_id', Integer, ForeignKey('directors.id'), primary_key=True)
)

movie_companies = Table(
    'movie_companies',
    Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id'), primary_key=True),
    Column('company_id', Integer, ForeignKey('production_companies.id'), primary_key=True)
)

class Movie(Base):
    """Movie model with comprehensive metadata"""
    __tablename__ = 'movies'
    
    id = Column(Integer, primary_key=True)  # TMDB movie ID
    title = Column(String(255), nullable=False, index=True)
    original_title = Column(String(255))
    overview = Column(Text)
    tagline = Column(Text)
    release_date = Column(String(10))  # YYYY-MM-DD format
    runtime = Column(Integer)  # minutes
    budget = Column(BigInteger)
    revenue = Column(BigInteger)
    vote_average = Column(Float)
    vote_count = Column(Integer)
    popularity = Column(Float)
    adult = Column(Boolean, default=False)
    video = Column(Boolean, default=False)
    
    # Status and production
    status = Column(String(50))  # Released, Post Production, etc.
    original_language = Column(String(10))
    homepage = Column(String(500))
    imdb_id = Column(String(20))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional metadata as JSON
    keywords = Column(JSON)
    spoken_languages = Column(JSON)
    production_countries = Column(JSON)
    
    # Relationships
    genres = relationship("Genre", secondary=movie_genres, back_populates="movies")
    actors = relationship("Actor", secondary=movie_actors, back_populates="movies")
    directors = relationship("Director", secondary=movie_directors, back_populates="movies")
    companies = relationship("ProductionCompany", secondary=movie_companies, back_populates="movies")
    reviews = relationship("Review", back_populates="movie")
    user_ratings = relationship("UserRating", back_populates="movie")

class Genre(Base):
    """Movie genres"""
    __tablename__ = 'genres'
    
    id = Column(Integer, primary_key=True)  # TMDB genre ID
    name = Column(String(100), nullable=False, unique=True, index=True)
    
    movies = relationship("Movie", secondary=movie_genres, back_populates="genres")

class Actor(Base):
    """Movie actors/cast"""
    __tablename__ = 'actors'
    
    id = Column(Integer, primary_key=True)  # TMDB person ID
    name = Column(String(255), nullable=False, index=True)
    gender = Column(Integer)  # 0: Not specified, 1: Female, 2: Male, 3: Non-binary
    known_for_department = Column(String(100))
    profile_path = Column(String(255))
    popularity = Column(Float)
    
    movies = relationship("Movie", secondary=movie_actors, back_populates="actors")

class Director(Base):
    """Movie directors"""
    __tablename__ = 'directors'
    
    id = Column(Integer, primary_key=True)  # TMDB person ID
    name = Column(String(255), nullable=False, index=True)
    gender = Column(Integer)
    known_for_department = Column(String(100))
    profile_path = Column(String(255))
    popularity = Column(Float)
    
    movies = relationship("Movie", secondary=movie_directors, back_populates="directors")

class ProductionCompany(Base):
    """Production companies"""
    __tablename__ = 'production_companies'
    
    id = Column(Integer, primary_key=True)  # TMDB company ID
    name = Column(String(255), nullable=False, index=True)
    logo_path = Column(String(255))
    origin_country = Column(String(10))
    
    movies = relationship("Movie", secondary=movie_companies, back_populates="companies")

class Review(Base):
    """Movie reviews from TMDB"""
    __tablename__ = 'reviews'
    
    id = Column(String(50), primary_key=True)  # TMDB review ID
    movie_id = Column(Integer, ForeignKey('movies.id'), nullable=False)
    author = Column(String(255))
    author_details = Column(JSON)  # Rating, avatar, etc.
    content = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    url = Column(String(500))
    
    movie = relationship("Movie", back_populates="reviews")

class UserRating(Base):
    """User ratings for movies (for future features)"""
    __tablename__ = 'user_ratings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)  # User identifier
    movie_id = Column(Integer, ForeignKey('movies.id'), nullable=False)
    rating = Column(Float, nullable=False)  # 1-10 scale
    review = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    movie = relationship("Movie", back_populates="user_ratings")

class SearchHistory(Base):
    """Track user search queries for analytics"""
    __tablename__ = 'search_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255))
    query = Column(Text, nullable=False)
    response_time = Column(Float)  # seconds
    results_count = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Default to SQLite for development
            database_url = os.getenv('DATABASE_URL', 'sqlite:///movies.db')
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def load_movies_from_json(self, json_file: str = "movies_dataset.json"):
        """Load movies from JSON file into database"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                movies_data = data.get('movies', [])
            
            session = self.get_session()
            
            print(f"🗃️ Loading {len(movies_data)} movies into database...")
            
            for movie_data in movies_data:
                try:
                    # Check if movie already exists
                    existing_movie = session.query(Movie).filter(Movie.id == movie_data['id']).first()
                    if existing_movie:
                        print(f"⏭️ Movie {movie_data['title']} already exists, skipping...")
                        continue
                    
                    # Create movie object
                    movie = Movie(
                        id=movie_data['id'],
                        title=movie_data.get('title', ''),
                        original_title=movie_data.get('original_title', ''),
                        overview=movie_data.get('overview', ''),
                        tagline=movie_data.get('tagline', ''),
                        release_date=movie_data.get('release_date', ''),
                        runtime=movie_data.get('runtime'),
                        budget=movie_data.get('budget', 0),
                        revenue=movie_data.get('revenue', 0),
                        vote_average=movie_data.get('vote_average', 0.0),
                        vote_count=movie_data.get('vote_count', 0),
                        popularity=movie_data.get('popularity', 0.0),
                        adult=movie_data.get('adult', False),
                        video=movie_data.get('video', False),
                        status=movie_data.get('status', ''),
                        original_language=movie_data.get('original_language', ''),
                        homepage=movie_data.get('homepage', ''),
                        imdb_id=movie_data.get('imdb_id', ''),
                        keywords=movie_data.get('keywords', {}),
                        spoken_languages=movie_data.get('spoken_languages', []),
                        production_countries=movie_data.get('production_countries', [])
                    )
                    
                    session.add(movie)
                    session.flush()  # Get the movie ID
                    
                    # Process genres
                    for genre_data in movie_data.get('genres', []):
                        genre = session.query(Genre).filter(Genre.id == genre_data['id']).first()
                        if not genre:
                            genre = Genre(id=genre_data['id'], name=genre_data['name'])
                            session.add(genre)
                        movie.genres.append(genre)
                    
                    # Process cast
                    cast_data = movie_data.get('credits', {}).get('cast', [])
                    for i, actor_data in enumerate(cast_data[:10]):  # Limit to top 10 actors
                        actor = session.query(Actor).filter(Actor.id == actor_data['id']).first()
                        if not actor:
                            actor = Actor(
                                id=actor_data['id'],
                                name=actor_data.get('name', ''),
                                gender=actor_data.get('gender'),
                                known_for_department=actor_data.get('known_for_department', ''),
                                profile_path=actor_data.get('profile_path', ''),
                                popularity=actor_data.get('popularity', 0.0)
                            )
                            session.add(actor)
                        movie.actors.append(actor)
                    
                    # Process directors
                    crew_data = movie_data.get('credits', {}).get('crew', [])
                    for crew_member in crew_data:
                        if crew_member.get('job') == 'Director':
                            director = session.query(Director).filter(Director.id == crew_member['id']).first()
                            if not director:
                                director = Director(
                                    id=crew_member['id'],
                                    name=crew_member.get('name', ''),
                                    gender=crew_member.get('gender'),
                                    known_for_department=crew_member.get('known_for_department', ''),
                                    profile_path=crew_member.get('profile_path', ''),
                                    popularity=crew_member.get('popularity', 0.0)
                                )
                                session.add(director)
                            movie.directors.append(director)
                    
                    # Process production companies
                    for company_data in movie_data.get('production_companies', []):
                        company = session.query(ProductionCompany).filter(ProductionCompany.id == company_data['id']).first()
                        if not company:
                            company = ProductionCompany(
                                id=company_data['id'],
                                name=company_data.get('name', ''),
                                logo_path=company_data.get('logo_path', ''),
                                origin_country=company_data.get('origin_country', '')
                            )
                            session.add(company)
                        movie.companies.append(company)
                    
                    print(f"✅ Added movie: {movie.title} ({movie.release_date[:4] if movie.release_date else 'Unknown'})")
                    
                except Exception as e:
                    print(f"❌ Error processing movie {movie_data.get('title', 'Unknown')}: {e}")
                    session.rollback()
                    continue
            
            session.commit()
            session.close()
            
            print(f"🎉 Successfully loaded movies into database!")
            
        except Exception as e:
            print(f"❌ Error loading movies into database: {e}")
            raise
    
    def get_movie_by_id(self, movie_id: int) -> Optional[Movie]:
        """Get movie by TMDB ID"""
        session = self.get_session()
        movie = session.query(Movie).filter(Movie.id == movie_id).first()
        session.close()
        return movie
    
    def search_movies(self, 
                     title: str = None, 
                     genre: str = None, 
                     year: int = None,
                     min_rating: float = None,
                     max_rating: float = None,
                     director: str = None,
                     actor: str = None,
                     limit: int = 20) -> List[Movie]:
        """Advanced movie search with multiple filters"""
        session = self.get_session()
        
        query = session.query(Movie)
        
        # Filter by title
        if title:
            query = query.filter(Movie.title.ilike(f'%{title}%'))
        
        # Filter by genre
        if genre:
            query = query.join(Movie.genres).filter(Genre.name.ilike(f'%{genre}%'))
        
        # Filter by year
        if year:
            query = query.filter(Movie.release_date.like(f'{year}%'))
        
        # Filter by rating
        if min_rating:
            query = query.filter(Movie.vote_average >= min_rating)
        if max_rating:
            query = query.filter(Movie.vote_average <= max_rating)
        
        # Filter by director
        if director:
            query = query.join(Movie.directors).filter(Director.name.ilike(f'%{director}%'))
        
        # Filter by actor
        if actor:
            query = query.join(Movie.actors).filter(Actor.name.ilike(f'%{actor}%'))
        
        # Add eager loading and order by popularity and limit
        from sqlalchemy.orm import joinedload
        movies = query.options(
            joinedload(Movie.genres),
            joinedload(Movie.actors),
            joinedload(Movie.directors)
        ).order_by(Movie.popularity.desc()).limit(limit).all()
        session.close()
        
        return movies
    
    def get_movie_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        session = self.get_session()
        
        stats = {
            'total_movies': session.query(Movie).count(),
            'total_genres': session.query(Genre).count(),
            'total_actors': session.query(Actor).count(),
            'total_directors': session.query(Director).count(),
            'avg_rating': session.query(func.avg(Movie.vote_average)).filter(Movie.vote_average > 0).scalar() or 0,
            'total_reviews': session.query(Review).count(),
            'total_searches': session.query(SearchHistory).count()
        }
        
        session.close()
        return stats
    
    def log_search(self, user_id: str, query: str, response_time: float, results_count: int):
        """Log search query for analytics"""
        session = self.get_session()
        
        search_log = SearchHistory(
            user_id=user_id,
            query=query,
            response_time=response_time,
            results_count=results_count
        )
        
        session.add(search_log)
        session.commit()
        session.close()
    
    def get_popular_searches(self, limit: int = 10) -> List[Dict]:
        """Get most popular search queries"""
        session = self.get_session()
        
        from sqlalchemy import func
        
        popular_queries = session.query(
            SearchHistory.query,
            func.count(SearchHistory.query).label('count')
        ).group_by(SearchHistory.query)\
         .order_by(func.count(SearchHistory.query).desc())\
         .limit(limit).all()
        
        session.close()
        
        return [{'query': q[0], 'count': q[1]} for q in popular_queries]

def main():
    """Test database functionality"""
    print("🗃️ Initializing Movie Database...")
    
    db = DatabaseManager()
    
    # Load movies from JSON
    if os.path.exists("movies_dataset.json"):
        db.load_movies_from_json()
    else:
        print("❌ No movies_dataset.json found. Run data_collector.py first.")
        return
    
    # Test queries
    print("\n🔍 Testing database queries...")
    
    # Get stats
    stats = db.get_movie_stats()
    print(f"📊 Database Stats: {stats}")
    
    # Search movies
    sci_fi_movies = db.search_movies(genre="Science Fiction", min_rating=7.0, limit=5)
    print(f"\n🚀 Found {len(sci_fi_movies)} high-rated sci-fi movies:")
    for movie in sci_fi_movies:
        print(f"  • {movie.title} ({movie.release_date[:4] if movie.release_date else 'Unknown'}) - ⭐ {movie.vote_average}")
    
    # Search by director
    nolan_movies = db.search_movies(director="Nolan", limit=5)
    print(f"\n🎬 Found {len(nolan_movies)} Nolan movies:")
    for movie in nolan_movies:
        directors = [d.name for d in movie.directors]
        print(f"  • {movie.title} - Directed by {', '.join(directors)}")

if __name__ == "__main__":
    main()

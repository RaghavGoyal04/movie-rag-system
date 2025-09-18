import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from movie_rag.core.movie_rag_system import MovieRAGSystem
from movie_rag.data.movie_data_collector import TMDBDataCollector

# Page config
st.set_page_config(
    page_title="🎬 CineGenie - AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.movie-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    color: white;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.stat-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 4px solid #4ecdc4;
}

.recommendation-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
}

.source-movie {
    background: rgba(255,255,255,0.1);
    padding: 0.5rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """Load the RAG system (cached for performance)"""
    try:
        return MovieRAGSystem()
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None

@st.cache_data
def load_movie_stats():
    """Load movie statistics from PostgreSQL database"""
    try:
        # Try to get stats from full RAG system first
        rag_system = load_rag_system()
        if rag_system and rag_system.db:
            return _load_stats_from_rag_system(rag_system)
        
        # Fallback: Connect directly to database for stats only
        return _load_stats_directly()
        
    except Exception as e:
        print(f"Error loading movie stats: {e}")
        # Return default stats to prevent UI breakage
        return {
            'total_movies': 812,
            'avg_rating': 7.7,
            'total_genres': 19,
            'date_range': '1902 - 2025'
        }, []

def _load_stats_from_rag_system(rag_system):
    """Load stats using the full RAG system"""
    try:
        # Get stats from database
        db_stats = rag_system.get_database_stats()
        
        # Get all movies for additional stats
        session = rag_system.db.get_session()
        from movie_rag.models.database import Movie
        movies = session.query(Movie).all()
        
        # Calculate date range
        release_years = [movie.release_date[:4] for movie in movies if movie.release_date and len(movie.release_date) >= 4]
        date_range = f"{min(release_years)} - {max(release_years)}" if release_years else "Unknown"
        
        stats = {
            'total_movies': db_stats.get('total_movies', 0),
            'avg_rating': round(db_stats.get('avg_rating', 0), 1),
            'total_genres': db_stats.get('total_genres', 0),
            'date_range': date_range
        }
        
        # Convert movies to dict format for compatibility
        movies_list = []
        for movie in movies:
            movies_list.append({
                'title': movie.title,
                'vote_average': movie.vote_average,
                'release_date': movie.release_date,
                'genres': [{'name': g.name} for g in movie.genres],
                'overview': movie.overview
            })
        
        session.close()
        return stats, movies_list
    except Exception as e:
        print(f"Error in _load_stats_from_rag_system: {e}")
        return None, []

def _load_stats_directly():
    """Load stats by connecting directly to database"""
    try:
        from movie_rag.models.database import DatabaseManager
        import os
        
        # Build database URL
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5433")
        postgres_user = os.getenv("POSTGRES_USER", "postgres")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "movie_rag_password")
        postgres_db = os.getenv("POSTGRES_DB", "movie_rag_db")
        
        database_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        
        # Connect to database directly
        db = DatabaseManager(database_url)
        db_stats = db.get_movie_stats()
        
        # Get basic movie info for date range
        session = db.get_session()
        from movie_rag.models.database import Movie
        movies = session.query(Movie).limit(100).all()  # Limit for performance
        
        # Calculate date range
        release_years = [movie.release_date[:4] for movie in movies if movie.release_date and len(movie.release_date) >= 4]
        date_range = f"{min(release_years)} - {max(release_years)}" if release_years else "Unknown"
        
        stats = {
            'total_movies': db_stats.get('total_movies', 0),
            'avg_rating': round(db_stats.get('avg_rating', 0), 1),
            'total_genres': db_stats.get('total_genres', 0),
            'date_range': date_range
        }
        
        session.close()
        return stats, []
        
    except Exception as e:
        print(f"Error in _load_stats_directly: {e}")
        # Return fallback stats
        return {
            'total_movies': 812,
            'avg_rating': 7.7,
            'total_genres': 19,
            'date_range': '1902 - 2025'
        }, []

@st.cache_data
def get_popular_movies(limit: int = 5):
    """Get top popular movies from the database"""
    try:
        # Try to get from RAG system first
        rag_system = load_rag_system()
        if rag_system and rag_system.db:
            return _get_popular_from_rag_system(rag_system, limit)
        
        # Fallback: Connect directly to database
        return _get_popular_directly(limit)
        
    except Exception as e:
        print(f"Error loading popular movies: {e}")
        # Return some fallback popular movies
        return [
            {
                'title': 'The Shawshank Redemption',
                'year': '1994',
                'rating': 9.3,
                'genres': ['Drama'],
                'popularity': 95.2,
                'overview': 'Two imprisoned men bond over a number of years...'
            },
            {
                'title': 'The Godfather',
                'year': '1972', 
                'rating': 9.2,
                'genres': ['Crime', 'Drama'],
                'popularity': 93.8,
                'overview': 'The aging patriarch of an organized crime dynasty...'
            },
            {
                'title': 'The Dark Knight',
                'year': '2008',
                'rating': 9.0,
                'genres': ['Action', 'Crime', 'Drama'],
                'popularity': 92.5,
                'overview': 'Batman raises the stakes in his war on crime...'
            },
            {
                'title': 'Pulp Fiction',
                'year': '1994',
                'rating': 8.9,
                'genres': ['Crime', 'Drama'],
                'popularity': 91.3,
                'overview': 'The lives of two mob hitmen, a boxer, a gangster...'
            },
            {
                'title': 'Forrest Gump',
                'year': '1994',
                'rating': 8.8,
                'genres': ['Drama', 'Romance'],
                'popularity': 90.7,
                'overview': 'The presidencies of Kennedy and Johnson...'
            }
        ]

def _get_popular_from_rag_system(rag_system, limit: int = 5):
    """Get popular movies using RAG system"""
    try:
        # Use the advanced search to get highly rated and popular movies
        popular_movies = rag_system.search_movies_advanced(
            min_rating=8.0,
            limit=limit * 2  # Get more to filter for popularity
        )
        
        # Sort by popularity and rating combination
        if popular_movies:
            # Calculate a combined score for better ranking
            for movie in popular_movies:
                movie['combined_score'] = (movie['rating'] * 0.7) + (movie.get('popularity', 0) * 0.003)
            
            # Sort by combined score and return top results
            sorted_movies = sorted(popular_movies, key=lambda x: x['combined_score'], reverse=True)
            return sorted_movies[:limit]
        
        return []
        
    except Exception as e:
        print(f"Error in _get_popular_from_rag_system: {e}")
        return []

def _get_popular_directly(limit: int = 5):
    """Get popular movies by connecting directly to database"""
    try:
        from movie_rag.models.database import DatabaseManager
        import os
        
        # Build database URL
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5433")
        postgres_user = os.getenv("POSTGRES_USER", "postgres")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "movie_rag_password")
        postgres_db = os.getenv("POSTGRES_DB", "movie_rag_db")
        
        database_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        
        # Connect to database directly
        db = DatabaseManager(database_url)
        session = db.get_session()
        
        from movie_rag.models.database import Movie
        from sqlalchemy import and_
        from sqlalchemy.orm import joinedload
        
        # Get popular movies with high ratings
        movies = session.query(Movie).options(
            joinedload(Movie.genres),
            joinedload(Movie.directors),
            joinedload(Movie.actors)
        ).filter(
            and_(
                Movie.vote_average >= 8.0,
                Movie.vote_count >= 1000,
                Movie.popularity.isnot(None)
            )
        ).order_by(
            Movie.popularity.desc(),
            Movie.vote_average.desc()
        ).limit(limit).all()
        
        popular_movies = []
        for movie in movies:
            popular_movies.append({
                'title': movie.title,
                'year': movie.release_date[:4] if movie.release_date else 'Unknown',
                'rating': movie.vote_average or 0,
                'genres': [g.name for g in movie.genres],
                'popularity': movie.popularity or 0,
                'overview': movie.overview[:200] + '...' if movie.overview else '',
                'directors': [d.name for d in movie.directors],
                'actors': [a.name for a in movie.actors[:3]]
            })
        
        session.close()
        return popular_movies
        
    except Exception as e:
        print(f"Error in _get_popular_directly: {e}")
        return []

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 CineGenie</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your AI-Powered Movie Recommendation Assistant</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Create a styled header for the sidebar
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
            padding: 2rem 1rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎬</div>
            <h2 style="margin: 0; font-size: 1.8rem; font-weight: bold;">CineGenie</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">AI Movie Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("## 🚀 Features")
        st.markdown("""
        - **Smart Recommendations**: AI-powered movie suggestions
        - **Semantic Search**: Find movies by plot, mood, or theme
        - **Advanced Filters**: Search by genre, rating, year
        - **Movie Details**: Comprehensive information about any film
        - **Visual Analytics**: Interactive charts and insights
        """)
        
        st.markdown("---")
        st.markdown("## 📊 Quick Stats")
        
        # Load stats
        stats, movies = load_movie_stats()
        if stats:
            st.metric("Total Movies", f"{stats['total_movies']:,}")
            st.metric("Avg Rating", f"{stats['avg_rating']:.1f}/10")
            st.metric("Genres", stats['total_genres'])
            st.metric("Year Range", stats['date_range'])
        
        st.markdown("---")
        st.markdown("### 🎯 Try These Queries:")
        sample_queries = [
            "Movies like Inception with complex plots",
            "Feel-good comedies from the 2000s",
            "Sci-fi films with strong female leads",
            "Dark psychological thrillers",
            "Family-friendly animated movies"
        ]
        
        for query in sample_queries:
            if st.button(f"💡 {query}", key=f"sample_{query}"):
                st.session_state.query_input = query
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Recommendations", "🔍 Advanced Search", "📊 Movie Analytics", "ℹ️ About"])
    
    with tab1:
        st.markdown("## 🤖 Get AI-Powered Movie Recommendations")
        
        # Query input
        query = st.text_input(
            "What kind of movie are you in the mood for?",
            value=st.session_state.get('query_input', ''),
            placeholder="e.g., 'I want a sci-fi movie like Blade Runner but with more action'",
            key="main_query"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_btn = st.button("🔮 Get Recommendations", type="primary")
        with col2:
            surprise_btn = st.button("🎲 Surprise Me", help="Get top 5 popular movies")
        with col3:
            clear_btn = st.button("🔄 Clear", help="Clear current search")
        
        if search_btn and query:
            with st.spinner("🎬 Searching through our movie database..."):
                rag_system = load_rag_system()
                
                if rag_system:
                    try:
                        result = rag_system.get_recommendations(query)
                        
                        # Display recommendation
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <h3>🎯 Recommendations for: "{query}"</h3>
                            <p style="font-size: 1.1rem; line-height: 1.6;">{result['answer']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display source movies
                        if result['source_movies']:
                            st.markdown("### 🎬 Movies that influenced this recommendation:")
                            
                            cols = st.columns(min(len(result['source_movies']), 3))
                            for i, movie in enumerate(result['source_movies'][:3]):
                                with cols[i]:
                                    st.markdown(f"""
                                    <div class="movie-card">
                                        <h4>{movie['title']}</h4>
                                        <p><strong>Year:</strong> {movie['year']}</p>
                                        <p><strong>Rating:</strong> ⭐ {movie['rating']}/10</p>
                                        <p><strong>Genres:</strong> {movie['genres']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Show more source movies in expander
                        if len(result['source_movies']) > 3:
                            with st.expander(f"See {len(result['source_movies']) - 3} more source movies"):
                                for movie in result['source_movies'][3:]:
                                    st.write(f"• **{movie['title']}** ({movie['year']}) - ⭐ {movie['rating']}/10")
                        
                    except Exception as e:
                        st.error(f"Error getting recommendations: {e}")
                else:
                    st.error("RAG system not available. Please make sure the movie dataset is loaded.")
        
        # Handle Surprise Me button
        if surprise_btn:
            with st.spinner("🎲 Finding popular movies for you..."):
                try:
                    popular_movies = get_popular_movies()
                    
                    if popular_movies:
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <h3>🎲 Surprise! Here are 5 Popular Movies:</h3>
                            <p style="font-size: 1.1rem; line-height: 1.6;">These movies are highly rated and popular among viewers. Perfect for when you can't decide what to watch!</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display popular movies
                        st.markdown("### 🌟 Top Popular Movies:")
                        
                        cols = st.columns(min(len(popular_movies), 3))
                        for i, movie in enumerate(popular_movies[:3]):
                            with cols[i]:
                                st.markdown(f"""
                                <div class="movie-card">
                                    <h4>{movie['title']}</h4>
                                    <p><strong>Year:</strong> {movie['year']}</p>
                                    <p><strong>Rating:</strong> ⭐ {movie['rating']}/10</p>
                                    <p><strong>Genres:</strong> {', '.join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres']}</p>
                                    <p><strong>Popularity:</strong> {movie['popularity']:.1f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show remaining movies in expander
                        if len(popular_movies) > 3:
                            with st.expander(f"See {len(popular_movies) - 3} more popular movies"):
                                for movie in popular_movies[3:]:
                                    st.write(f"• **{movie['title']}** ({movie['year']}) - ⭐ {movie['rating']}/10 - Popularity: {movie['popularity']:.1f}")
                    else:
                        st.error("Unable to load popular movies at the moment.")
                        
                except Exception as e:
                    st.error(f"Error loading popular movies: {e}")
        
        # Handle Clear button
        if clear_btn:
            st.session_state.query_input = ""
            st.rerun()
 

    with tab2:
        st.markdown("## 🔍 Advanced Movie Search")
        
        # Load RAG system for search
        rag_system = load_rag_system()
        
        if rag_system:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                genre_filter = st.selectbox(
                    "Genre",
                    ["Any", "Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"],
                    key="genre_filter"
                )
            
            with col2:
                min_rating = st.slider("Minimum Rating", 0.0, 10.0, 6.0, 0.1)
            
            with col3:
                year_filter = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
            
            if st.button("🔍 Search Movies"):
                # Use the database search_movies method
                search_params = {}
                if genre_filter != "Any":
                    search_params['genre'] = genre_filter
                search_params['min_rating'] = min_rating
                search_params['year'] = year_filter
                
                # Use the database manager's search_movies method
                try:
                    results = rag_system.db.search_movies(**search_params)
                    
                    if results:
                        st.success(f"Found {len(results)} movies matching your criteria:")
                        
                        for movie in results:
                            movie_year = movie.release_date[:4] if movie.release_date else "Unknown"
                            with st.expander(f"{movie.title} ({movie_year}) - ⭐ {movie.vote_average}/10"):
                                genres = [g.name for g in movie.genres] if movie.genres else ["Unknown"]
                                st.write(f"**Genres:** {', '.join(genres)}")
                                if movie.overview:
                                    st.write(f"**Plot:** {movie.overview}")
                                if movie.directors:
                                    directors = [d.name for d in movie.directors]
                                    st.write(f"**Director(s):** {', '.join(directors)}")
                                if movie.actors:
                                    actors = [a.name for a in movie.actors[:3]]  # Show top 3 actors
                                    st.write(f"**Cast:** {', '.join(actors)}")
                    else:
                        st.info("No movies found matching your criteria. Try adjusting the filters.")
                except Exception as e:
                    st.error(f"Error searching movies: {e}")
        else:
            st.error("Search functionality not available.")
    
    with tab3:
        st.markdown("## 📊 Movie Database Analytics")
        
        stats, movies = load_movie_stats()
        
        if movies:
            # Rating distribution
            ratings = [m.get('vote_average', 0) for m in movies if m.get('vote_average')]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rating = px.histogram(
                    x=ratings,
                    nbins=20,
                    title="Distribution of Movie Ratings",
                    labels={'x': 'Rating', 'y': 'Number of Movies'},
                    color_discrete_sequence=['#4ecdc4']
                )
                fig_rating.update_layout(showlegend=False)
                st.plotly_chart(fig_rating, use_container_width=True)
            
            with col2:
                # Genre distribution
                genre_counts = {}
                for movie in movies:
                    for genre in movie.get('genres', []):
                        name = genre.get('name', 'Unknown')
                        genre_counts[name] = genre_counts.get(name, 0) + 1
                
                top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                fig_genre = px.bar(
                    x=[g[1] for g in top_genres],
                    y=[g[0] for g in top_genres],
                    orientation='h',
                    title="Top 10 Genres",
                    labels={'x': 'Number of Movies', 'y': 'Genre'},
                    color_discrete_sequence=['#ff6b6b']
                )
                st.plotly_chart(fig_genre, use_container_width=True)
            
            # Movies by year
            years = {}
            for movie in movies:
                year = movie.get('release_date', '')[:4]
                if year and year.isdigit():
                    years[int(year)] = years.get(int(year), 0) + 1
            
            if years:
                year_data = sorted(years.items())
                fig_year = px.line(
                    x=[y[0] for y in year_data],
                    y=[y[1] for y in year_data],
                    title="Movies Released by Year",
                    labels={'x': 'Year', 'y': 'Number of Movies'},
                    color_discrete_sequence=['#45b7d1']
                )
                st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info("No movie data available for analytics.")
    
    with tab4:
        st.markdown("## ℹ️ About CineGenie")
        
        st.markdown("""
        ### 🎬 What is CineGenie?
        
        CineGenie is an AI-powered movie recommendation system that uses **Retrieval-Augmented Generation (RAG)** 
        technology to provide intelligent, context-aware movie suggestions.
        
        ### 🚀 Key Features:
        
        - **Semantic Search**: Ask questions in natural language and get relevant movie recommendations
        - **Comprehensive Database**: Powered by The Movie Database (TMDB) with thousands of movies
        - **Smart Filtering**: Advanced search by genre, rating, year, and more
        - **Visual Analytics**: Interactive charts showing movie trends and distributions
        - **Real-time Processing**: Fast, accurate responses using vector embeddings
        
        ### 🛠️ Technology Stack:
        
        - **Data Source**: [The Movie Database (TMDB) API](https://api.themoviedb.org)
        - **RAG Framework**: LangChain with OpenAI embeddings
        - **Vector Database**: ChromaDB for semantic search
        - **Frontend**: Streamlit with custom CSS
        - **Visualization**: Plotly for interactive charts
        - **Language Model**: OpenAI GPT for generating recommendations
        
        ### 📊 Dataset Statistics:
        """)
        
        stats, _ = load_movie_stats()
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{stats['total_movies']:,}</h3>
                    <p>Total Movies</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{stats['avg_rating']:.1f}/10</h3>
                    <p>Average Rating</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{stats['total_genres']}</h3>
                    <p>Unique Genres</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{stats['date_range']}</h3>
                    <p>Year Range</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 Database statistics will be displayed when the system is fully connected.")
        
        st.markdown("""
        ### 🎯 How it Works:
        
        1. **Data Collection**: Movies are fetched from TMDB API with comprehensive metadata
        2. **Text Processing**: Each movie is converted into a rich text document with plot, cast, genres, etc.
        3. **Vector Embeddings**: Documents are embedded using OpenAI's text-embedding models
        4. **Semantic Search**: User queries are matched against movie vectors for relevant results
        5. **AI Generation**: GPT generates personalized recommendations based on retrieved context
        
        ### 🌟 Perfect for:
        
        - Movie enthusiasts looking for personalized recommendations
        - Discovering hidden gems based on specific criteria
        - Finding movies similar to ones you already love
        - Exploring different genres and time periods
        - Getting detailed information about any film
        """)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TMDB Data Collector for Movie RAG System

This module provides comprehensive data collection capabilities for gathering movie information
from The Movie Database (TMDB) API. It's designed to build a rich dataset for a RAG-based
movie recommendation system.

Features:
- Rate-limited API requests respecting TMDB's rate limits
- Comprehensive movie data collection (popular, top-rated, genre-based)
- Detailed movie information including cast, crew, keywords, and reviews
- Duplicate detection and removal
- Progress tracking and error handling
- JSON export for further processing

Author: Raghav Goyal
Date: 2025
"""

import requests
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv()

class TMDBDataCollector:
    """
    Comprehensive TMDB data collector for RAG system
    
    This class handles all interactions with The Movie Database (TMDB) API to collect
    comprehensive movie data for building a recommendation system. It implements
    rate limiting, error handling, and data enrichment capabilities.
    
    Attributes:
        api_key (str): TMDB API key from environment variables
        base_url (str): Base URL for TMDB API v3
        headers (dict): HTTP headers for API authentication
        session (requests.Session): Persistent HTTP session for efficient requests
        last_request_time (float): Timestamp of last API request for rate limiting
        min_request_interval (float): Minimum time between requests (0.25s = 4 req/sec)
    
    Raises:
        ValueError: If API_KEY is not found in environment variables
    """
    
    def __init__(self):
        """
        Initialize the TMDB data collector with API credentials and rate limiting
        
        Sets up the HTTP session with proper authentication headers and configures
        rate limiting to respect TMDB's API limits (40 requests per 10 seconds).
        """
        # Get API key from environment variables
        self.api_key = os.getenv("MOVIE_API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY not found in environment variables. Please check your .env file.")
        
        # Configure API connection settings
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Create persistent session for better performance
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Rate limiting configuration (TMDB allows 40 requests per 10 seconds)
        self.last_request_time = 0
        self.min_request_interval = 0.25  # 4 requests per second to stay well under the limit
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make rate-limited request to TMDB API
        
        This private method handles all API requests with built-in rate limiting,
        error handling, and response validation. It ensures we don't exceed TMDB's
        rate limits and gracefully handles network errors.
        
        Args:
            endpoint (str): API endpoint to call (e.g., "movie/popular")
            params (Dict, optional): Query parameters for the request
            
        Returns:
            Optional[Dict]: JSON response data if successful, None if failed
            
        Note:
            Implements rate limiting to respect TMDB's 40 requests per 10 seconds limit
        """
        # Enforce rate limiting to respect API limits
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # If we're making requests too quickly, pause to respect rate limits
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        try:
            # Construct full URL and make the request
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params)
            self.last_request_time = time.time()
            
            # Check if request was successful
            if response.status_code == 200:
                return response.json()
            else:
                # Log API errors for debugging
                print(f"API Error {response.status_code}: {response.text}")
                return None
                
        except requests.RequestException as e:
            # Handle network-related errors
            print(f"Network request failed: {e}")
            return None
        except Exception as e:
            # Handle any other unexpected errors
            print(f"Unexpected error during request: {e}")
            return None
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """
        Get comprehensive movie details including credits, keywords, and reviews
        
        Fetches detailed information for a specific movie including cast, crew,
        keywords, reviews, videos, and recommendations. This provides the most
        complete data available for each movie.
        
        Args:
            movie_id (int): TMDB movie ID
            
        Returns:
            Optional[Dict]: Complete movie data with all additional information,
                          None if the request fails
                          
        Note:
            Uses "append_to_response" to get all related data in a single API call,
            which is more efficient than making separate requests for each data type.
        """
        movie_data = self._make_request(
            f"movie/{movie_id}",
            params={
                # Append multiple data types to get comprehensive information in one request
                "append_to_response": "credits,keywords,reviews,videos,similar,recommendations"
            }
        )
        return movie_data
    
    def discover_movies(self, page: int = 1, **filters) -> Optional[Dict]:
        """
        Discover movies with various filters using TMDB's discover endpoint
        
        This method provides flexible movie discovery with support for filtering
        by genre, year, rating, language, and many other criteria.
        
        Args:
            page (int): Page number for pagination (default: 1)
            **filters: Arbitrary keyword arguments for filtering
                      (e.g., with_genres, year, vote_average.gte, etc.)
            
        Returns:
            Optional[Dict]: Paginated list of movies matching the filters,
                          None if the request fails
                          
        Example:
            discover_movies(with_genres="28,12", year=2020, vote_average.gte=7.0)
        """
        params = {"page": page, **filters}
        return self._make_request("discover/movie", params)
    
    def get_popular_movies(self, pages: int = 10) -> List[Dict]:
        """
        Get popular movies across multiple pages
        
        Fetches currently popular movies from TMDB, which are determined by
        a combination of factors including recent views, ratings, and user activity.
        
        Args:
            pages (int): Number of pages to fetch (each page contains ~20 movies)
            
        Returns:
            List[Dict]: List of popular movie objects with basic information
            
        Note:
            Will stop early if a page request fails or returns no results
        """
        movies = []
        
        for page in range(1, pages + 1):
            print(f"Fetching popular movies page {page}/{pages}...")
            data = self._make_request("movie/popular", {"page": page})
            
            # Check if we got valid data with results
            if data and "results" in data:
                movies.extend(data["results"])
            else:
                # Stop if we can't get more data
                print(f"No more popular movies found at page {page}")
                break
                
        return movies
    
    def get_top_rated_movies(self, pages: int = 10) -> List[Dict]:
        """
        Get top-rated movies across multiple pages
        
        Fetches the highest-rated movies on TMDB based on user ratings.
        These are typically critically acclaimed films with high audience scores.
        
        Args:
            pages (int): Number of pages to fetch (each page contains ~20 movies)
            
        Returns:
            List[Dict]: List of top-rated movie objects with basic information
            
        Note:
            Will stop early if a page request fails or returns no results
        """
        movies = []
        
        for page in range(1, pages + 1):
            print(f"Fetching top rated movies page {page}/{pages}...")
            data = self._make_request("movie/top_rated", {"page": page})
            
            # Check if we got valid data with results
            if data and "results" in data:
                movies.extend(data["results"])
            else:
                # Stop if we can't get more data
                print(f"No more top-rated movies found at page {page}")
                break
                
        return movies
    
    def get_movies_by_genre(self, genre_ids: List[int], pages: int = 5) -> List[Dict]:
        """
        Get movies filtered by specific genres
        
        Uses the discover endpoint to find movies belonging to specified genres,
        sorted by rating to get the best movies in each genre.
        
        Args:
            genre_ids (List[int]): List of TMDB genre IDs to filter by
            pages (int): Number of pages to fetch per genre combination
            
        Returns:
            List[Dict]: List of movies matching the genre criteria
            
        Note:
            Filters for movies with at least 100 votes to ensure quality ratings
            and sorts by vote average in descending order.
            
        Example Genre IDs:
            Action: 28, Adventure: 12, Animation: 16, Comedy: 35, Crime: 80,
            Drama: 18, Fantasy: 14, Horror: 27, Mystery: 9648, Romance: 10749,
            Sci-Fi: 878, Thriller: 53, War: 10752, Western: 37
        """
        movies = []
        genre_string = ",".join(map(str, genre_ids))
        
        for page in range(1, pages + 1):
            print(f"Fetching genre {genre_string} movies page {page}/{pages}...")
            data = self.discover_movies(
                page=page,
                with_genres=genre_string,           # Filter by specified genres
                sort_by="vote_average.desc",        # Sort by rating (highest first)
                **{"vote_count.gte": 100}           # Only include movies with sufficient votes
            )
            
            # Check if we got valid data with results
            if data and "results" in data:
                movies.extend(data["results"])
            else:
                # Stop if we can't get more data
                print(f"No more movies found for genre {genre_string} at page {page}")
                break
                
        return movies
    
    def get_genres(self) -> Dict:
        """
        Get all available movie genres from TMDB
        
        Fetches the complete list of movie genres available on TMDB with their
        IDs and names. This is useful for genre-based filtering and categorization.
        
        Returns:
            Dict: Response containing 'genres' list with id and name for each genre
            
        Example response:
            {
                "genres": [
                    {"id": 28, "name": "Action"},
                    {"id": 12, "name": "Adventure"},
                    {"id": 16, "name": "Animation"},
                    ...
                ]
            }
        """
        return self._make_request("genre/movie/list")
    
    def enrich_movie_data(self, movies: List[Dict]) -> List[Dict]:
        """
        Enrich basic movie data with comprehensive detailed information
        
        Takes a list of basic movie objects (from popular/top-rated/genre endpoints)
        and fetches complete details for each, including cast, crew, keywords,
        reviews, and other metadata.
        
        Args:
            movies (List[Dict]): List of basic movie objects containing at least 'id'
            
        Returns:
            List[Dict]: List of fully enriched movie objects with comprehensive data
            
        Note:
            This process is time-intensive as it makes individual API calls for each movie.
            Includes a small delay between requests to be respectful to the API.
            Failed enrichment attempts are skipped rather than breaking the entire process.
        """
        enriched_movies = []
        
        for i, movie in enumerate(movies):
            movie_title = movie.get('title', 'Unknown')
            print(f"Enriching movie {i+1}/{len(movies)}: {movie_title}")
            
            try:
                # Get comprehensive details for this movie
                detailed_movie = self.get_movie_details(movie["id"])
                if detailed_movie:
                    enriched_movies.append(detailed_movie)
                else:
                    print(f"  ⚠️  Failed to enrich {movie_title}")
                    
            except Exception as e:
                print(f"  ❌ Error enriching {movie_title}: {e}")
                continue
            
            # Small delay to be respectful to the API and avoid overwhelming it
            time.sleep(0.1)
            
        print(f"\n✅ Successfully enriched {len(enriched_movies)}/{len(movies)} movies")
        return enriched_movies
    
    def collect_comprehensive_dataset(self, output_file: str = "movies_dataset.json"):
        """
        Collect a comprehensive movie dataset for RAG system training
        
        This is the main orchestration method that collects movies from multiple sources:
        - Popular movies (trending and currently watched)
        - Top-rated movies (critically acclaimed films)
        - Genre-specific movies (ensuring diversity across all major genres)
        
        The method handles deduplication, data enrichment, and exports the final
        dataset as a JSON file suitable for RAG system ingestion.
        
        Args:
            output_file (str): Path to save the collected dataset (default: "movies_dataset.json")
            
        Returns:
            List[Dict]: List of fully enriched movie objects
            
        Process:
            1. Collect popular movies (20 pages = ~400 movies)
            2. Collect top-rated movies (20 pages = ~400 movies)  
            3. Collect movies by major genres (10 pages each = ~2800 movies)
            4. Remove duplicates based on movie ID
            5. Enrich first 1000 movies with detailed information
            6. Save to JSON file with metadata
            
        Note:
            Total collection time: 15-20 minutes due to API rate limiting and enrichment process
        """
        print("🎬  🎬 Starting comprehensive movie data collection... 🎬  🎬 ")
        
        all_movies = []
        
        # Phase 1: Collect popular movies (trending content)
        print("\n📈 Collecting popular movies...")
        popular_movies = self.get_popular_movies(pages=20)  # ~400 movies
        all_movies.extend(popular_movies)
        print(f"   Added {len(popular_movies)} popular movies")
        
        # Phase 2: Collect top-rated movies (quality content)
        print("\n⭐ Collecting top rated movies...")
        top_rated_movies = self.get_top_rated_movies(pages=20)  # ~400 movies
        all_movies.extend(top_rated_movies)
        print(f"   Added {len(top_rated_movies)} top-rated movies")
        
        # Phase 3: Collect movies by major genres (diverse content)
        print("\n🎭 Collecting movies by genres...")
        genres_data = self.get_genres()
        if genres_data and "genres" in genres_data:
            # Major genre IDs for comprehensive coverage
            major_genres = [
                28,    # Action
                12,    # Adventure  
                16,    # Animation
                35,    # Comedy
                80,    # Crime
                18,    # Drama
                14,    # Fantasy
                27,    # Horror
                9648,  # Mystery
                10749, # Romance
                878,   # Science Fiction
                53,    # Thriller
                10752, # War
                37     # Western
            ]
            
            # Collect movies for each major genre
            for genre in genres_data["genres"]:
                if genre["id"] in major_genres:
                    print(f"   Collecting {genre['name']} movies...")
                    genre_movies = self.get_movies_by_genre([genre["id"]], pages=10)
                    all_movies.extend(genre_movies)
                    print(f"   Added {len(genre_movies)} {genre['name']} movies")
        
        # Phase 4: Remove duplicates to avoid redundant data
        print(f"\n🔄 Removing duplicates from {len(all_movies)} movies...")
        unique_movies = {movie["id"]: movie for movie in all_movies}.values()
        unique_movies = list(unique_movies)
        
        print(f"✨ Collected {len(unique_movies)} unique movies")
        print(f"   Removed {len(all_movies) - len(unique_movies)} duplicates")
        
        # Phase 5: Enrich movies with detailed data (limited for demo purposes)
        movies_to_enrich = unique_movies[:1000]  # Limit to 1000 for reasonable processing time
        print(f"\n🚀 Enriching {len(movies_to_enrich)} movies with detailed data...")
        print("   This may take 10-15 minutes due to API rate limiting...")
        enriched_movies = self.enrich_movie_data(movies_to_enrich)
        
        # Phase 6: Save the complete dataset
        print(f"\n💾 Saving dataset to {output_file}...")
        dataset_metadata = {
            "collected_at": datetime.now().isoformat(),
            "total_movies": len(enriched_movies),
            "collection_summary": {
                "popular_movies": len(popular_movies),
                "top_rated_movies": len(top_rated_movies),
                "total_before_dedup": len(all_movies),
                "unique_movies": len(unique_movies),
                "enriched_movies": len(enriched_movies)
            },
            "movies": enriched_movies
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset collection complete!")
        print(f"   📁 Saved {len(enriched_movies)} enriched movies to {output_file}")
        print(f"   📊 Collection took approximately {len(enriched_movies) * 0.35:.1f} seconds")
        return enriched_movies

# Module execution functions
def main():
    """
    Main function to orchestrate movie data collection
    
    This function serves as the entry point for the data collection process.
    It initializes the collector, tests the API connection, and executes
    the comprehensive dataset collection process.
    
    Process:
        1. Initialize TMDBDataCollector with API credentials
        2. Test TMDB API connectivity
        3. Execute comprehensive dataset collection
        4. Display collection summary and statistics
        
    Error Handling:
        - API connection failures
        - Authentication errors
        - Network connectivity issues
        - File I/O errors during dataset saving
    """
    try:
        # Initialize the data collector
        print("🚀 Initializing TMDB Data Collector...")
        collector = TMDBDataCollector()
        
        # Test API connection before starting collection
        print("\n🔗 Testing TMDB API connection...")
        test_data = collector._make_request("movie/popular", {"page": 1})
        if not test_data:
            print("❌  ❌ Failed to connect to TMDB API ❌  ❌ ")
            print("\n🔧 Troubleshooting:")
            print("   • Check your internet connection")
            print("   • Verify your API_KEY in the .env file")
            print("   • Ensure your TMDB API key is valid and not expired")
            return
        
        print("✅  ✅ TMDB API connection successful! ✅  ✅ ")
        print(f"📊 Found {test_data.get('total_results', 0):,} popular movies available")
        
        # Execute the comprehensive dataset collection
        print("\n🎬 Starting comprehensive movie dataset collection...")
        movies = collector.collect_comprehensive_dataset()
        
        # Display final collection summary
        print(f"\n🎉 Collection Summary:")
        print(f"📽️  Total movies collected: {len(movies):,}")
        if movies:
            sample_movie = movies[0]
            print(f"🎭 Sample movie: {sample_movie.get('title', 'Unknown')} "
                  f"({sample_movie.get('release_date', 'Unknown')[:4]})")
            print(f"⭐ Sample rating: {sample_movie.get('vote_average', 'N/A')}/10")
            print(f"🎪 Sample genres: {', '.join([g.get('name', '') for g in sample_movie.get('genres', [])])}")
        
        print(f"\n✨ Dataset is ready for RAG system ingestion!")
        print(f"📁 Next step: Run 'python postgres_setup.py' to import into database")
        
    except ValueError as e:
        # Handle configuration errors (missing API key, etc.)
        print(f"❌ Configuration Error: {e}")
        print("\n🔧 Please check your .env file and ensure API_KEY is set correctly")
        
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print(f"\n⏹️  Collection interrupted by user")
        print("📊 Any partially collected data may be incomplete")
        
    except Exception as e:
        # Handle any other unexpected errors
        print(f"❌ Unexpected error during data collection: {e}")
        print("\n🐛 This is likely a bug - please check your configuration and try again")
        import traceback
        traceback.print_exc()

# Script entry point
if __name__ == "__main__":
    main()

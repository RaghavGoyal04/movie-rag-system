"""
Movie RAG System - A comprehensive movie recommendation system using RAG (Retrieval-Augmented Generation)

This package provides:
- Data collection from TMDB API
- PostgreSQL database integration
- ChromaDB vector storage
- Enhanced RAG-based recommendations
- Streamlit web interface
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.movie_rag_enhanced import MovieRAGSystem
from .models.database import DatabaseManager
from .data.movie_data_collector import TMDBDataCollector

__all__ = [
    "MovieRAGSystem",
    "DatabaseManager", 
    "TMDBDataCollector"
]

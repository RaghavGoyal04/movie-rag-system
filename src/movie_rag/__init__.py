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
__author__ = "Raghav Goyal"
__email__ = "gyl.rghv@gmail.com"

from .core.movie_rag_system import MovieRAGSystem
from .models.database import DatabaseManager
from .data.movie_data_collector import TMDBDataCollector

__all__ = [
    "MovieRAGSystem",
    "DatabaseManager", 
    "TMDBDataCollector"
]

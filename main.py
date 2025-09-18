#!/usr/bin/env python3
"""
Main entry point for the Movie RAG System
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from movie_rag.core.movie_rag_enhanced import main

if __name__ == "__main__":
    print("🎬 Starting Movie RAG System...")
    main()

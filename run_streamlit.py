#!/usr/bin/env python3
"""
Streamlit app runner for the Movie RAG System
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import streamlit as st
    
    app_path = src_path / "movie_rag" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())

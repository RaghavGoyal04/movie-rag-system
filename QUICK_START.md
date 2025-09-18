# 🚀 Quick Start Guide

## Setup in 5 Steps

### 1. Environment Setup
```bash
# Create .env file with your API keys
python setup_env.py

# Edit .env file with your actual keys:
# - GITHUB_TOKEN (from GitHub settings)
# - API_KEY (from TMDB)
```

### 2. Start Services
```bash
# Start PostgreSQL + ChromaDB
docker-compose up -d

# Setup database (first time only)
uv run scripts/postgres_setup.py
```

### 3. Test Everything
```bash
# Run end-to-end test
uv run test_e2e.py

# Or run individual tests
uv run tests/test_integration.py
```

### 4. Run the Application
```bash
# CLI interface
uv run main.py

# Web interface
uv run run_streamlit.py
```

### 5. Access Web Interface
- **Movie App**: http://localhost:8501
- **pgAdmin**: http://localhost:5050 (admin@admin.com / admin)
- **ChromaDB**: http://localhost:8000

## API Keys Required

### GitHub Models (Recommended)
```bash
GITHUB_TOKEN=ghp_your_token_here
```
- Get from: https://github.com/settings/tokens
- Free tier available
- Models: gpt-4o-mini, text-embedding-3-small

### TMDB API
```bash
API_KEY=your_tmdb_key_here
```
- Get from: https://www.themoviedb.org/settings/api
- Free tier available
- For movie data collection

## Project Structure
```
movie-rag-system/
├── main.py                   # 🚀 CLI entry point
├── run_streamlit.py          # 🌐 Web entry point
├── test_e2e.py              # 🧪 End-to-end test
├── setup_env.py             # ⚙️ Environment setup
├── src/movie_rag/           # 📦 Main package
├── tests/                   # 🧪 Test suite
├── scripts/                 # 📋 Setup scripts
└── docs/                    # 📚 Documentation
```

## Troubleshooting

**Services not running?**
```bash
docker-compose up -d
```

**Database empty?**
```bash
uv run scripts/postgres_setup.py
```

**API errors?**
- Check your .env file
- Verify API key permissions
- Check API quotas

**Import errors?**
```bash
uv sync
```

## Next Steps
- Add more movies: Edit `scripts/postgres_setup.py`
- Customize models: Edit `.env` file
- Add features: Extend `src/movie_rag/`

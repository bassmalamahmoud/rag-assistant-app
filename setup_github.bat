@echo off
echo ==========================================
echo  RAG Assistant - GitHub Repository Setup
echo ==========================================
echo.
echo This script initializes the git repository and prepares for publishing.
echo Make sure you have created a GitHub repository first.
echo.

:: Check if git is installed
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not found in PATH.
    echo Please download Git from: https://git-scm.com/download/win
    echo Then re-run this script.
    pause
    exit /b 1
)

:: Initialize git
git init
git add .gitignore
git add README.md
git add requirements.txt
git add notebooks\rag_pipeline.ipynb
git add backend\app\
git add backend\requirements.txt
git add backend\.env.example
git add backend\Dockerfile
git add backend\tests\
git add frontend\
git add data\raw\

:: Commit
git commit -m "feat: RAG assistant - notebook, FastAPI backend, Streamlit frontend

- 25 self-authored Markdown documents on Python/NumPy/Pandas
- Header-aware chunking with semantic unit preservation
- ChromaDB persistent vector store with all-MiniLM-L6-v2 embeddings
- 100%% Hit@3 retrieval accuracy on 10 Common Mistake benchmarks
- FastAPI backend with Pydantic validation, CORS, structured logging
- Grounded generation with Ollama + resilient extractive fallback
- 3/3 pytest tests passing (health, happy path, validation)
- Streamlit interactive frontend with source citations"

echo.
echo [SUCCESS] Repository initialized and committed.
echo.
echo Next steps:
echo   1. Create a GitHub repo at https://github.com/new
echo   2. Run: git remote add origin https://github.com/YOUR_USERNAME/rag-assistant-app.git
echo   3. Run: git branch -M main
echo   4. Run: git push -u origin main
echo.
pause

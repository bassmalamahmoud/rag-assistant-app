@echo off
echo ==========================================
echo  RAG Assistant - Streamlit Frontend
echo ==========================================
echo Ensure the FastAPI backend is running first on port 8000!
echo Starting Streamlit frontend on http://localhost:8501
echo Press Ctrl+C to stop.
echo.
cd /d %~dp0
if exist "..\.venv\Scripts\streamlit.exe" (
    ..\.venv\Scripts\streamlit.exe run app.py
) else if exist "E:\rag_venv\Scripts\streamlit.exe" (
    E:\rag_venv\Scripts\streamlit.exe run app.py
) else (
    streamlit run app.py
)

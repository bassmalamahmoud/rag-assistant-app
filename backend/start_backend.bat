@echo off
echo ==========================================
echo  RAG Assistant - Backend Server
echo ==========================================
echo Ensure the virtual environment is set up.
echo Starting FastAPI backend on http://localhost:8000
echo Press Ctrl+C to stop.
echo.
cd /d %~dp0
if exist "..\.venv\Scripts\uvicorn.exe" (
    ..\.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload
) else if exist "E:\rag_venv\Scripts\uvicorn.exe" (
    E:\rag_venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload
) else (
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
)

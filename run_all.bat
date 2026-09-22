@echo off
echo ========================================================
echo   Launching Python for ML RAG Assistant (Full Stack)
echo ========================================================
echo.
echo Starting FastAPI backend in a separate window...
start "RAG Backend (FastAPI)" cmd /c "cd /d %~dp0backend && call start_backend.bat"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo Starting Streamlit frontend in a separate window...
start "RAG Frontend (Streamlit)" cmd /c "cd /d %~dp0frontend && call start_frontend.bat"

echo.
echo ========================================================
echo  System is booting!
echo  Backend:  http://localhost:8000 (API docs at /docs)
echo  Frontend: http://localhost:8501
echo ========================================================
echo.
pause

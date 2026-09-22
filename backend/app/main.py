from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import query
from app.services.retrieval import retrieval_service
from app.utils.logging_config import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG Assistant API...")
    try:
        retrieval_service.initialize()
        logger.info("Retrieval service initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not fully initialize vector store on startup: {e}. Will retry on query.")
    yield
    logger.info("Shutting down RAG Assistant API...")

app = FastAPI(
    title="RAG-Powered Document Assistant API",
    description="Production FastAPI backend serving grounded RAG queries on Python for Machine Learning (Core Python, NumPy, Pandas)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from Streamlit frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(query.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import APIRouter, HTTPException, status
from app.schemas.query import QueryRequest, QueryResponse
from app.services.retrieval import retrieval_service
from app.services.generation import generation_service
from app.utils.logging_config import logger

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "ok",
        "service": "RAG-Powered Document Assistant API",
        "version": "1.0.0"
    }

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query_rag(request: QueryRequest):
    logger.info(f"Received query: '{request.question}'")
    try:
        documents, metadatas = retrieval_service.retrieve(request.question)
        answer = generation_service.generate_answer(request.question, documents, metadatas)
        
        # Deduplicate sources preserving order
        sources = list(dict.fromkeys([m.get("source", "unknown.md") for m in metadatas if "source" in m]))
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the query: {str(e)}"
        )

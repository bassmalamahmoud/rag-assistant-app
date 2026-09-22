from pydantic import BaseModel, Field, field_validator

class QueryRequest(BaseModel):
    question: str = Field(..., description="The technical question to query the RAG assistant about", min_length=1)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or solely whitespace.")
        return v.strip()

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The synthesized grounded answer")
    sources: list[str] = Field(default_factory=list, description="List of source document filenames retrieved")

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

class AIDetectionRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=15000, description="Input text to detect AI vs Human authorship")

    @field_validator("text")
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace.")
        return v

class AIDetectionResponse(BaseModel):
    text_snippet: str
    is_ai: bool
    label: str
    ai_probability: float
    human_probability: float
    confidence_score: float
    confidence_level: str
    assessment: str
    word_count: Optional[int] = None
    min_reliable_words: Optional[int] = 50
    is_short_text: Optional[bool] = False
    reliability_warning: Optional[str] = None
    execution_time_seconds: Optional[float] = None

class EditDiff(BaseModel):
    action: str  # 'replace', 'delete', 'insert'
    original_tokens: str
    corrected_tokens: str

class SentenceCorrection(BaseModel):
    original: str
    corrected: str
    has_errors: bool
    error_count: int
    edits: List[EditDiff]

class GrammarRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=15000, description="Input text to check and correct grammar")

    @field_validator("text")
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace.")
        return v

class GrammarResponse(BaseModel):
    original_text: str
    corrected_text: str
    has_errors: bool
    total_errors: int
    sentence_corrections: List[SentenceCorrection]
    execution_time_seconds: Optional[float] = None

class CombinedAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=15000)
    enable_ai_detection: bool = True
    enable_grammar: bool = True

    @field_validator("text")
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace.")
        return v

class TextStatistics(BaseModel):
    char_count: int
    word_count: int
    sentence_count: int

class CombinedAnalysisResponse(BaseModel):
    input_text: str
    statistics: TextStatistics
    ai_detection: Optional[AIDetectionResponse] = None
    grammar_analysis: Optional[GrammarResponse] = None
    execution_time_seconds: float

class HealthResponse(BaseModel):
    status: str
    version: str
    ai_detector_loaded: bool
    grammar_model_loaded: bool
    device: str
    timestamp: str

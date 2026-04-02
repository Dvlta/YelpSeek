from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)


class RestaurantResult(BaseModel):
    business_id: str
    name: str
    city: str
    stars: float
    review_count: int
    categories: str
    address: str
    similarity_score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    results: list[RestaurantResult]
    latency_ms: int
    total_results: int

import time

from fastapi import APIRouter, Request

from models.schemas import SearchRequest, SearchResponse, RestaurantResult

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, body: SearchRequest):
    start = time.time()
    retriever = request.app.state.retriever
    raw_results = retriever.search(body.query, top_k=body.top_k)
    latency_ms = round((time.time() - start) * 1000)

    results = [RestaurantResult(**r) for r in raw_results]

    return SearchResponse(
        query=body.query,
        results=results,
        latency_ms=latency_ms,
        total_results=len(results),
    )

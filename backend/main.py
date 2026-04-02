import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import search
from services.retriever import Retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = Retriever(
        index_path=os.getenv("INDEX_PATH", "data/index.faiss"),
        metadata_path=os.getenv("METADATA_PATH", "data/metadata.parquet"),
        model_path=os.getenv("MODEL_PATH", "models/encoder_v1/best"),
    )
    yield


app = FastAPI(title="YelpSeek API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(search.router)

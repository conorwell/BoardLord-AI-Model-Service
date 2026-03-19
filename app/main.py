from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import predictor
from app.routers import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="Board Lord Grade Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

app.include_router(predict.router)

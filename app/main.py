from contextlib import asynccontextmanager
from fastapi import FastAPI
from app import predictor
from app.routers import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="Board Lord Grade Predictor", lifespan=lifespan)
app.include_router(predict.router)

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes.query import (
    router as query_router,
    neo4j_service,
    table_index_service,
    graph_rag_service,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s", settings.app_title)
    neo4j_service.set_services(table_index_service, graph_rag_service)
    await table_index_service.build_index()
    logger.info("Table index built on startup")
    logger.info("Docs available at http://localhost:8000/docs")
    yield
    logger.info("Shutting down %s", settings.app_title)


app = FastAPI(
    title=settings.app_title,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api/v1")

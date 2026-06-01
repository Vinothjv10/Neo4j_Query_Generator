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
    embedding_service,
    dail_sql_service,
    gnn_schema_service,
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

    # Wire core services
    neo4j_service.set_services(table_index_service, graph_rag_service)
    table_index_service.set_embedding_service(embedding_service)

    # Build TF-IDF + embedding index (existing, ~18s)
    await table_index_service.build_index()
    logger.info("Table index built")

    # Build GNN schema graph (fast — pure Neo4j Cypher + NetworkX, no GPU)
    try:
        await gnn_schema_service.build_graph()
        logger.info("GNN schema graph built")
    except Exception as exc:
        logger.warning("GNN schema graph build failed (non-fatal): %s", exc)

    # Initialise DAIL-SQL SQLite store
    try:
        await dail_sql_service.initialize()
        count = await dail_sql_service.example_count()
        logger.info("DAIL-SQL example store ready (%d examples)", count)
    except Exception as exc:
        logger.warning("DAIL-SQL store init failed (non-fatal): %s", exc)

    logger.info("All services ready. Docs: http://localhost:8000/docs")
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

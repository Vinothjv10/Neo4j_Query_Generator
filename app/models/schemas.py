from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 100


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    description: str | None = None


class TableInfo(BaseModel):
    table_name: str
    schema_name: str
    description: str | None = None
    columns: list[ColumnInfo]
    related_tables: list[str]


class SchemaContext(BaseModel):
    tables: list[TableInfo]


class TableReason(BaseModel):
    table: str
    tier: str
    description: str
    top_columns: list[str]
    reason: str


class ColumnReason(BaseModel):
    column: str
    score: float
    reason: str


class Reasoning(BaseModel):
    table_selection: list[TableReason]
    column_selection: dict[str, list[ColumnReason]]
    final_explanation: str | None = None
    sql_generation: str
    retries: list[str]
    # Advanced features
    agent_mode: bool = False
    agent_trace: list[str] = []
    few_shot_count: int = 0
    gnn_boost_applied: bool = False


class QueryResponse(BaseModel):
    question: str
    generated_sql: str
    results: list[dict]
    row_count: int
    schema_tables_used: list[str]
    reasoning: Reasoning | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None

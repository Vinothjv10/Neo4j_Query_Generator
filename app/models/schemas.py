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


class QueryResponse(BaseModel):
    question: str
    generated_sql: str
    results: list[dict]
    row_count: int
    schema_tables_used: list[str]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None

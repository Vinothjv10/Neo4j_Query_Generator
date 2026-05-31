from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    postgres_dsn: str = "postgresql://user:password@localhost:5432/dbname"

    google_application_credentials: str = ""
    vertex_ai_project: str = ""
    vertex_ai_location: str = ""
    vertex_ai_model: str = ""

    app_title: str = "Text2SQL BI API"
    debug: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

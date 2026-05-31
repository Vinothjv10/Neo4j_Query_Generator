import os
import json

from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx

from app.config import settings
from app.utils.logger import log_step

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class LLMService:
    def __init__(self):
        self._token: str | None = None
        self._credentials = None

    async def _ensure_token(self) -> str:
        if self._token:
            return self._token

        creds_path = settings.google_application_credentials
        if not creds_path or not os.path.exists(creds_path):
            raise FileNotFoundError(
                f"Service account file not found: {creds_path}. "
                "Set GOOGLE_APPLICATION_CREDENTIALS in .env"
            )

        abs_path = os.path.abspath(creds_path)
        self._credentials = service_account.Credentials.from_service_account_file(
            abs_path,
            scopes=SCOPES,
        )

        auth_req = GoogleAuthRequest()
        self._credentials.refresh(auth_req)
        self._token = self._credentials.token

        project = settings.vertex_ai_project
        if not project:
            with open(abs_path) as f:
                project = json.load(f).get("project_id", "")

        log_step("GCP", "Vertex AI credentials loaded",
                 project=project,
                 location=settings.vertex_ai_location,
                 model=settings.vertex_ai_model)
        return self._token

    async def generate_sql(self, system_prompt: str, user_prompt: str) -> str:
        token = await self._ensure_token()

        project = settings.vertex_ai_project
        location = settings.vertex_ai_location
        model = settings.vertex_ai_model

        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project}/locations/{location}/"
            f"publishers/anthropic/models/{model}:rawPredict"
        )

        log_step("LLM", f"Calling Claude on Vertex AI",
                 model=model, max_tokens=2048, temperature=0.10)

        payload = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "system": system_prompt,
            "max_tokens": 2048,
            "temperature": 0.10,
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise ValueError(
                f"Vertex AI API error (HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )

        data = response.json()
        content_blocks = data.get("content", [])
        raw = ""
        for block in content_blocks:
            if block.get("type") == "text":
                raw += block.get("text", "")

        raw = raw.strip()
        log_step("LLM", f"Raw response received", chars=len(raw))

        if raw == "UNABLE_TO_GENERATE":
            raise ValueError(
                "The LLM could not generate a SQL query for this question "
                "with the available schema."
            )

        sql = self._clean_sql(raw)
        log_step("LLM", f"SQL after cleaning", sql=sql.replace("\n", " "))

        if not sql:
            raise ValueError("LLM returned an empty or invalid response.")

        return sql

    def _clean_sql(self, raw: str) -> str:
        if raw.startswith("```"):
            lines = raw.splitlines()
            cleaned: list[str] = []
            for line in lines:
                if line.startswith("```"):
                    continue
                cleaned.append(line)
            return "\n".join(cleaned).strip()
        return raw
